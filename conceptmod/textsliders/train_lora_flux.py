# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List, Optional
import argparse
import ast
from pathlib import Path
import random
import gc

import copy
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np


from conceptmod.textsliders.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from conceptmod.textsliders.dora import DoRANetwork
from conceptmod.textsliders import train_util
from conceptmod.textsliders import model_util
from conceptmod.textsliders import prompt_util
from conceptmod.textsliders.prompt_util import (
    PromptEmbedsCache,
    PromptEmbedsPair,
    PromptSettings,
    PromptEmbedsSD3,
)
from conceptmod.textsliders import debug_util
from conceptmod.textsliders import config_util
from conceptmod.textsliders.config_util import RootConfig

import wandb
import yaml

NUM_IMAGES_PER_PROMPT = 1

def render_debug(fname, latents, pipeline):
    with torch.no_grad():
        latents = latents.float()
        pipeline.vae = pipeline.vae.float()
        pipeline.vae.decoder.cuda()
        latents = pipeline._unpack_latents(latents, 512, 512, pipeline.vae_scale_factor)
        latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor

        image = pipeline.vae.decode(latents, return_dict=False)[0]

        image = pipeline.image_processor.postprocess(image, output_type="png")
        image = Image.fromarray((image[0] * 255).astype(np.uint8))
        image.save(fname)

def flush():
    torch.cuda.empty_cache()
    gc.collect()


def calculate_memory_size(tensor):
    """Calculate memory size of a tensor in bytes."""
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()

def print_grad_memory_usage(model):
    total_memory = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_size = calculate_memory_size(param.grad)
            total_memory += grad_size
            print(f"Layer: {name} | Size: {param.size()} | Gradient Memory: {grad_size / 1e6:.2f} MB")
    
    print(f"Total Gradient Memory Usage: {total_memory / 1e6:.2f} MB")

def log_mem(msg):
    return
    print("log mem:", msg)
    # Check the memory allocated (used by tensors)
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Allocated Memory: {allocated_memory / 1024**2:.2f} MB")

    # Check the memory reserved (total reserved by the caching allocator)
    reserved_memory = torch.cuda.memory_reserved()
    print(f"Reserved Memory: {reserved_memory / 1024**2:.2f} MB")


def train(
    config: RootConfig,
    prompts: list[PromptSettings],
    device,
    on_step_complete,
    peft_type='lora',
    rank=4,
    save_file=True
):
    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    save_path = Path(config.save.path)

    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    if config.logging.verbose:
        print(metadata)

    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)
    print("Loading", config.pretrained_model.name_or_path)

    guidance_scale = config.train.cfg
    (
        tokenizers,
        text_encoders,
        transformer,
        noise_scheduler,
        pipeline
    ) = model_util.load_models_flux(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
        weight_dtype = weight_dtype
    )

    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    transformer.requires_grad_(False)
    transformer.eval()
    if peft_type == 'dora':
        peft_class = DoRANetwork
    else:
        peft_class = LoRANetwork

    target_replace="Attention"

    network = peft_class(
        transformer,
        rank=rank,
        multiplier=1.0,
        delimiter="-",
        target_replace=[target_replace],
        train_method=config.network.training_method
    ).to(device, dtype=weight_dtype)
    print("network", len(list(network.parameters())), "parameters")

    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    criteria = torch.nn.MSELoss()

    if config.logging.verbose:
        print("Prompts")
        for settings in prompts:
            print(settings)

    # debug
    if config.logging.verbose:
        debug_util.check_requires_grad(network)
        debug_util.check_training_mode(network)

    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    with torch.no_grad():
        for settings in prompts:
            if config.logging.verbose:
                print(settings)
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
                settings.negative
            ]:
                if cache[prompt] == None:
                    tex_embs, pool_embs = train_util.encode_prompts_flux(
                            pipeline,
                            tokenizers,
                            text_encoders,
                            prompt,
                            num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                        )
                    tex_embs.to("cpu")
                    pool_embs.to("cpu")
                    cache[prompt] = PromptEmbedsSD3(
                        tex_embs,
                        pool_embs
                    )

            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    cache[settings.negative],
                    settings,
                )
            )

    pipeline.tokenizer = None
    pipeline.tokenizer_2 = None
    pipeline.text_encoder = None
    pipeline.text_encoder_2 = None

    log_mem("before del tok")
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        del tokenizer, text_encoder

    del tokenizers, text_encoders

    flush()
    log_mem("after del tok")

    loss = None
    transformer.requires_grad_(False)
    if settings.batch_size < 8:
        accumulation_steps = 8 // settings.batch_size + (1 if (8 % settings.batch_size) > 0 else 0)
    else:
        accumulation_steps = 1
    pbar = tqdm(range(config.train.iterations*accumulation_steps+1))
    for i in pbar:
        with torch.no_grad():
            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]

            for embedding in prompt_pair.embeddings():
                embedding.pooled_embeds.to("cuda:0")
                embedding.text_embeds.to("cuda:0")

            #timesteps_to = random.choice([0,1])
            timesteps_to=0
            num_inference_steps = 8#timesteps_to + 1
            should_render_debug=False

            height, width = prompt_pair.resolution, prompt_pair.resolution
            if prompt_pair.dynamic_resolution:
                height, width = train_util.get_random_resolution_in_bucket(
                    prompt_pair.resolution
                )

            if config.logging.verbose:
                print("gudance_scale:", prompt_pair.guidance_scale)
                print("resolution:", prompt_pair.resolution)
                print("dynamic_resolution:", prompt_pair.dynamic_resolution)
                if prompt_pair.dynamic_resolution:
                    print("bucketed resolution:", (height, width))
                print("batch_size:", prompt_pair.batch_size)
                print("dynamic_crops:", prompt_pair.dynamic_crops)

            latents = train_util.get_initial_latents_flux(
                noise_scheduler, prompt_pair.batch_size, height, width, 1
            ).to(device, dtype=weight_dtype)

            denoised_latents = latents

            with network:
                denoised_latents, timesteps = train_util.diffusion_flux(
                        pipeline,
                        transformer,
                        noise_scheduler,
                        latents,
                        text_embeddings=prompt_pair.unconditional.text_embeds,
                        add_text_embeddings=prompt_pair.unconditional.pooled_embeds,
                        start_timesteps=0,
                        total_timesteps=timesteps_to,
                        guidance_scale=guidance_scale,
                        num_inference_steps = num_inference_steps 
                        )
                current_timestep = timesteps[timesteps_to]
                log_mem("after denoised_latents")

            step_index = noise_scheduler._step_index

            positive_latents = denoised_latents
            steps=1
            #for j in range(1 - timesteps_to):
            for j in range(steps):
                current_timestep = timesteps[timesteps_to + j]
                positive_latents = train_util.predict_noise_flux(
                        pipeline,
                        transformer,
                        noise_scheduler,
                        current_timestep,
                        positive_latents,
                        text_embeddings=prompt_pair.positive.text_embeds.clone(),
                        add_text_embeddings=prompt_pair.positive.pooled_embeds.clone(),
                        #text_embeddings=train_util.concat_embeddings_flux(
                        #    prompt_pair.unconditional.text_embeds,
                        #    prompt_pair.positive.text_embeds,
                        #    prompt_pair.batch_size,
                        #    ),
                        #add_text_embeddings=train_util.concat_embeddings_flux(
                        #    prompt_pair.unconditional.pooled_embeds,
                        #    prompt_pair.positive.pooled_embeds,
                        #    prompt_pair.batch_size,
                        #    ),
                        guidance_scale=guidance_scale,
                        ).to(device, dtype=weight_dtype)
            current_timestep = timesteps[timesteps_to]
            noise_scheduler._step_index=step_index
            log_mem("after positive_latents")
            for j in range(steps):
                current_timestep = timesteps[timesteps_to + j]
                neutral_latents = train_util.predict_noise_flux(
                        pipeline,
                        transformer,
                        noise_scheduler,
                        current_timestep,
                        denoised_latents,
                        text_embeddings=prompt_pair.neutral.text_embeds,
                        add_text_embeddings=prompt_pair.neutral.pooled_embeds,
                        guidance_scale=guidance_scale,
                        ).to(device, dtype=weight_dtype)

            noise_scheduler._step_index=step_index
            log_mem("after neutral_latents")

            negative_latents = denoised_latents
            #for j in range(1 - timesteps_to):
            for j in range(steps):
                current_timestep = timesteps[timesteps_to + j]
                negative_latents = train_util.predict_noise_flux(
                        pipeline,
                        transformer,
                        noise_scheduler,
                        current_timestep,
                        negative_latents,
                        text_embeddings=prompt_pair.negative.text_embeds,
                        add_text_embeddings=prompt_pair.negative.pooled_embeds,
                        guidance_scale=guidance_scale,
                        ).to(device, dtype=weight_dtype)
            current_timestep = timesteps[timesteps_to]
            log_mem("after negative_latents")

            if config.logging.verbose:
                print("positive_latents:", positive_latents[0, 0, :5, :5])
                print("neutral_latents:", neutral_latents[0, 0, :5, :5])

        log_mem("pre network")
        noise_scheduler._step_index=step_index
        with network:
            for j in range(steps):
                target_latents = train_util.predict_noise_flux(
                        pipeline,
                        transformer,
                        noise_scheduler,
                        current_timestep,
                        denoised_latents,
                        text_embeddings=prompt_pair.target.text_embeds,
                        add_text_embeddings=prompt_pair.target.pooled_embeds,
                        guidance_scale=guidance_scale,
                        ).to(device, dtype=weight_dtype)
            log_mem("after target_latents")
            if config.logging.verbose:
                print("target_latents:", target_latents[0, 0, :5, :5])

        positive_latents.requires_grad = False
        negative_latents.requires_grad = False
        neutral_latents.requires_grad = False

        log_mem("before loss")

        loss = prompt_pair.loss(
            target_latents=target_latents,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
            negative_latents=negative_latents,
        )

        torch.cuda.empty_cache()  # Optional: Force clearing of cache, though usually not needed

        log_mem("after loss")

        pbar.set_description(f"Loss*1k: {loss.item()*1000:.4f}")
        if config.logging.use_wandb:
            wandb.log(
                    {"loss": loss, "iteration": i, "lr": lr_scheduler.get_last_lr()[0]}
                    )

        if (i + 1) % accumulation_steps == 0:
            optimizer.zero_grad()  # Reset the gradients
        log_mem("before backward loss")
        if accumulation_steps > 1:
            loss = loss / accumulation_steps
        loss.backward()
        #print_grad_memory_usage(network)
        log_mem("after backward loss")

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_value_(network.parameters(), clip_value=1.0)
            optimizer.step()  # Update the model parameters
            lr_scheduler.step()

        if should_render_debug:
            render_debug(f"train{i}.png", target_latents, pipeline)
            render_debug(f"train{i}p.png", positive_latents, pipeline)
            render_debug(f"train{i}n.png", negative_latents, pipeline)
            #render_debug(f"train{i}fp.png", full_positive_latents, pipeline)
        del positive_latents, neutral_latents, negative_latents  # Free memory explicitly when done
        del (
                target_latents,
                denoised_latents,
                latents,
                )
        flush()
        
        if (
            i % config.save.per_steps == 0
            and i != 0
            and i != config.train.iterations - 1
            and save_file
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.safetensors",
                dtype=save_weight_dtype,
            )
        if (i + 1) % accumulation_steps == 0:
            if on_step_complete is not None:
                on_step_complete((i+1)//accumulation_steps)

    if save_file:
        print("Saving...",save_path / f"{config.save.name}_last.safetensors" )
        save_path.mkdir(parents=True, exist_ok=True)
        network.save_weights(
            save_path / f"{config.save.name}_last.safetensors",
            dtype=save_weight_dtype,
        )

        del (
            transformer,
            noise_scheduler,
            loss,
            optimizer,
            network,
        )

    else:
        return network.get_state_dict(save_weight_dtype)


def main(args):
    config_file = args.config_file

    config = config_util.load_config_from_yaml(config_file)
    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(',')
        attributes = [a.strip() for a in attributes]
    print("ATTRIBUTES", attributes)
    
    config.network.alpha = args.alpha
    config.network.rank = args.rank
    config.save.name += f'_alpha{args.alpha}'
    config.save.name += f'_rank{config.network.rank }'
    config.save.name += f'_{config.network.training_method}'
    config.save.path += f'/{config.save.name}'
    
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    if config.logging.verbose:
        print(prompts)
    device = torch.device(f"cuda:{args.device}")
    train(config, prompts, device, on_step_complete=None, rank=args.rank, peft_type=args.peft_type)

def train_lora(target, positive, negative, unconditional, alpha=1.0, device=0, name=None, attributes=None, batch_size=1, config_file='data/config-sd3.yaml', resolution=512, steps=None, on_step_complete=None, peft_type='lora', rank=4):
    # Create the configuration dictionary
    output_dict = {
        "target": target,
        "positive": positive,
        "negative": negative,
        "unconditional": unconditional,
        "neutral": target,  # Assuming neutral is the same as target
        "action": "enhance",
        "resolution": resolution,
        "dynamic_resolution": False,
        "batch_size": batch_size
    }

    # Writing the dictionary to 'data/prompts-sd3.yaml'
    with open('data/prompts-flux.yaml', 'w') as file:
        yaml.dump([output_dict], file)  # Note the list wrapping around output_dict

    config = config_util.load_config_from_yaml(config_file)
    if name is not None:
        config.save.name = name
    if steps is not None:
        config.train.iterations = steps
    attr_list = []
    if attributes is not None:
        attr_list = [a.strip() for a in attributes.split(',')]

    config.network.alpha = alpha
    config.network.rank = rank
    config.save.name += f'_alpha{alpha}'
    config.save.name += f'_rank{rank}'
    config.save.name += f'_{config.network.training_method}'
    config.save.path += f'/{config.save.name}'

    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attr_list)
    device = torch.device(f"cuda:{device}")
    return train(config, prompts, device, on_step_complete, save_file=False, rank=rank, peft_type=peft_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Config file for training.",
    )
    # config_file 'data/config.yaml'
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="LoRA weight.",
    )
    # --alpha 1.0
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=4,
    )
    # --rank 4
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to train on.",
    )
    # --device 0
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Device to train on.",
    )
    # --name 'eyesize_slider'
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default=None,
        help="attritbutes to disentangle (comma seperated string)",
    )
    parser.add_argument(
        "--peft_type",
        type=str,
        required=False,
        default="lora",
        help="dora or lora (default)",
    )
 
    args = parser.parse_args()

    main(args)
