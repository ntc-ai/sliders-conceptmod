# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List, Optional
import argparse
import ast
from pathlib import Path
import gc

import torch
from tqdm import tqdm


from conceptmod.textsliders.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from conceptmod.textsliders.dora import DoRANetwork
from conceptmod.textsliders import train_util
from conceptmod.textsliders import model_util
from conceptmod.textsliders import prompt_util
from conceptmod.textsliders.prompt_util import (
    PromptEmbedsCache,
    PromptEmbedsPair,
    PromptSettings,
    PromptEmbedsXL,
)
from conceptmod.textsliders import debug_util
from conceptmod.textsliders import config_util
from conceptmod.textsliders.config_util import RootConfig

import wandb
import yaml

NUM_IMAGES_PER_PROMPT = 1


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(
    config: RootConfig,
    prompts: list[PromptSettings],
    device,
    on_step_complete,
    peft_type='dora',
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
        unet,
        noise_scheduler,
    ) = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )

    for text_encoder in text_encoders:
        text_encoder.to(device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    if config.other.use_xformers:
        unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()
    if peft_type == 'dora':
        network = DoRANetwork(
            unet,
            rank=rank,
            multiplier=1.0,
            alpha=config.network.alpha,
            train_method=config.network.training_method,
        ).to(device, dtype=weight_dtype)

    else:
        network = LoRANetwork(
            unet,
            rank=rank,
            multiplier=1.0,
            alpha=config.network.alpha,
            train_method=config.network.training_method,
        ).to(device, dtype=weight_dtype)

    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    #optimizer_args
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
            
    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs)
    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
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
                    tex_embs, pool_embs = train_util.encode_prompts_xl(
                            tokenizers,
                            text_encoders,
                            [prompt],
                            num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                        )
                    cache[prompt] = PromptEmbedsXL(
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

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        del tokenizer, text_encoder

    flush()

    pbar = tqdm(range(config.train.iterations))

    loss = None

    for i in pbar:
        with torch.no_grad():
            noise_scheduler.set_timesteps(
                config.train.max_denoising_steps, device=device
            )

            optimizer.zero_grad()

            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]

            # 1 ~ 49 からランダム
            #timesteps_to = torch.randint(
            #    1, config.train.max_denoising_steps, (1,)
            #).item()
            timesteps_to = config.train.max_denoising_steps-1

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

            latents = train_util.get_initial_latents(
                noise_scheduler, prompt_pair.batch_size, height, width, 1
            ).to(device, dtype=weight_dtype)

            add_time_ids = train_util.get_add_time_ids(
                height,
                width,
                dynamic_crops=prompt_pair.dynamic_crops,
                dtype=weight_dtype,
            ).to(device, dtype=weight_dtype)

            with network:
                # ちょっとデノイズされれたものが返る
                denoised_latents = train_util.diffusion_xl(
                    unet,
                    noise_scheduler,
                    latents,  # 単純なノイズのlatentsを渡す
                    text_embeddings=train_util.concat_embeddings(
                        prompt_pair.unconditional.text_embeds,
                        prompt_pair.target.text_embeds,
                        prompt_pair.batch_size,
                    ),
                    add_text_embeddings=train_util.concat_embeddings(
                        prompt_pair.unconditional.pooled_embeds,
                        prompt_pair.target.pooled_embeds,
                        prompt_pair.batch_size,
                    ),
                    add_time_ids=train_util.concat_embeddings(
                        add_time_ids, add_time_ids, prompt_pair.batch_size
                    ),
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=guidance_scale
                ) #TODO: How does the gradient work?

            noise_scheduler.set_timesteps(1000)

            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / config.train.max_denoising_steps)
            ]

            # with network: の外では空のLoRAのみが有効になる
            positive_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.text_embeds,
                    prompt_pair.positive.text_embeds,
                    prompt_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.pooled_embeds,
                    prompt_pair.positive.pooled_embeds,
                    prompt_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids, add_time_ids, prompt_pair.batch_size
                ),
                guidance_scale=guidance_scale, #TODO
            ).to(device, dtype=weight_dtype)
            neutral_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.text_embeds,
                    prompt_pair.neutral.text_embeds,
                    prompt_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.pooled_embeds,
                    prompt_pair.neutral.pooled_embeds,
                    prompt_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids, add_time_ids, prompt_pair.batch_size
                ),
                guidance_scale=guidance_scale, #TODO
            ).to(device, dtype=weight_dtype)
            negative_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.text_embeds,
                    prompt_pair.negative.text_embeds,
                    prompt_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.pooled_embeds,
                    prompt_pair.negative.pooled_embeds,
                    prompt_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids, add_time_ids, prompt_pair.batch_size
                ),
                guidance_scale=guidance_scale, #TODO
            ).to(device, dtype=weight_dtype)

            if config.logging.verbose:
                print("positive_latents:", positive_latents[0, 0, :5, :5])
                print("neutral_latents:", neutral_latents[0, 0, :5, :5])

        with network:
            target_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.text_embeds,
                    prompt_pair.target.text_embeds,
                    prompt_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.pooled_embeds,
                    prompt_pair.target.pooled_embeds,
                    prompt_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids, add_time_ids, prompt_pair.batch_size
                ),
                guidance_scale=guidance_scale, #TODO
            ).to(device, dtype=weight_dtype)

            if config.logging.verbose:
                print("target_latents:", target_latents[0, 0, :5, :5])

        positive_latents.requires_grad = False
        negative_latents.requires_grad = False
        neutral_latents.requires_grad = False

        loss = prompt_pair.loss(
            target_latents=target_latents,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
            negative_latents=negative_latents,
        )

        # 1000倍しないとずっと0.000...になってしまって見た目的に面白くない
        pbar.set_description(f"Loss*1k: {loss.item()*1000:.4f}")
        if config.logging.use_wandb:
            wandb.log(
                {"loss": loss, "iteration": i, "lr": lr_scheduler.get_last_lr()[0]}
            )

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        del (
            positive_latents,
            negative_latents,
            neutral_latents,
            target_latents,
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
        if on_step_complete is not None:
            on_step_complete(i)

    if save_file:
        print("Saving...",save_path / f"{config.save.name}_last.safetensors" )
        save_path.mkdir(parents=True, exist_ok=True)
        network.save_weights(
            save_path / f"{config.save.name}_last.safetensors",
            dtype=save_weight_dtype,
        )

        del (
            unet,
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
    train(config, prompts, device, on_step_complete=None, rank=args.rank)

def train_lora(target, positive, negative, unconditional, alpha=1.0, device=0, name=None, attributes=None, batch_size=1, config_file='data/config-xl.yaml', resolution=512, steps=None, on_step_complete=None, peft_type='lora', rank=4):
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

    # Writing the dictionary to 'data/prompts-xl.yaml'
    with open('data/prompts-xl.yaml', 'w') as file:
        yaml.dump([output_dict], file)  # Note the list wrapping around output_dict

    #print("Data saved to 'data/prompts-xl.yaml'")
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
    return train(config, prompts, device, on_step_complete, save_file=False, rank=rank)

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
        default="dora",
        help="dora (default) or lora",
    )
 
    args = parser.parse_args()

    main(args)
