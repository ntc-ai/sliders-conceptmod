# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List, Optional
import argparse
import ast
from pathlib import Path
import gc
import numpy as np

import torch
from tqdm import tqdm


from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
import train_util
import model_util
import prompt_util
from prompt_util import (
    PromptEmbedsCache,
    PromptEmbedsPair,
    PromptSettings,
    PromptEmbedsXL,
)
import debug_util
import config_util
from config_util import RootConfig

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
    save_file=True,
    clip_index=0,
    positive=None,
    negative=None
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

    (
        tokenizers,
        text_encoders,
        unet,
        noise_scheduler,
    ) = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
        lora='',
        lora_weight=0
    )

    del unet
    flush()
    (
        tokenizers2,
        text_encoders2,
        unet2,
        noise_scheduler,
    ) = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
        lora='',
        lora_weight=0
    )
    del unet2
    flush()

    for text_encoder in text_encoders+text_encoders2:
        text_encoder.to(device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    index = clip_index
    prefix = ["lora_te1","lora_te2"][index]

    text_encoder = text_encoders[index]
    network = LoRANetwork(
        text_encoder,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        prefix=prefix,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)
    network.requires_grad_(True)

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

    settings = prompts[0]
    # debug
    if config.logging.verbose:
        debug_util.check_requires_grad(network)
        debug_util.check_training_mode(network)

    pbar = tqdm(range(config.train.iterations))

    print("INDEX", index)
    chosenlayer = -2

    for i in pbar:
        with network:
            for l in network.unet_loras:
                l.multiplier = 1.0
            tokens = train_util.text_tokenize(tokenizers[index], [''])

            neu_tex_embs = text_encoder(
                tokens.to(text_encoder.device), output_hidden_states=True
            ).hidden_states[chosenlayer]

            tokens = train_util.text_tokenize(tokenizers2[index], [", ".join(positive)])

            pos_tex_embs = text_encoders2[index](
                tokens.to(text_encoders2[index].device), output_hidden_states=True
            ).hidden_states[chosenlayer]

            loss = ((pos_tex_embs - neu_tex_embs) ** 2).mean()
            if negative is not None:
                for l in network.unet_loras:
                    l.multiplier = -1.0


                tokens = train_util.text_tokenize(tokenizers2[index], [", ".join(negative)])
                pos_tex_embs = text_encoders2[index](
                    tokens.to(text_encoders2[index].device), output_hidden_states=True
                ).hidden_states[chosenlayer]

                tokens = train_util.text_tokenize(tokenizers[index], [''])

                neu_tex_embs2 = text_encoder(
                    tokens.to(text_encoder.device), output_hidden_states=True
                ).hidden_states[chosenlayer]


                loss = ((pos_tex_embs - neu_tex_embs2) ** 2).mean()


        # 1000倍しないとずっと0.000...になってしまって見た目的に面白くない
        pbar.set_description(f"Loss*1k: {loss.item()*1000:.4f}")
        if config.logging.use_wandb:
            wandb.log(
                {"loss": loss, "iteration": i, "lr": lr_scheduler.get_last_lr()[0]}
            )

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if (
            i % config.save.per_steps == 0
            and i != 0
            and i != config.train.iterations - 1
            and save_file
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{index}_{i}steps.safetensors",
                dtype=save_weight_dtype,
            )
        if on_step_complete is not None:
            on_step_complete(i)

    if save_file:
        print("Saving...",save_path / f"{config.save.name}_last.safetensors" )
        save_path.mkdir(parents=True, exist_ok=True)
        network.save_weights(
            save_path / f"{config.save.name}_{index}_last.safetensors",
            dtype=save_weight_dtype,
        )

        del (
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
    train(config, prompts, device, on_step_complete=None, positive = args.positive, negative=args.negative, clip_index=args.clip_index)

def train_lora(target, positive, negative, unconditional, alpha=1.0, rank=4, device=0, name=None, attributes=None, batch_size=1, config_file='data/config-xl.yaml', resolution=512, steps=None, on_step_complete=None):
    # Create the configuration dictionary
    output_dict = {
        "target": target,
        "positive": positive,
        "negative": negative,
        "unconditional": unconditional,
        "neutral": target,  # Assuming neutral is the same as target
        "action": "enhance",
        "guidance_scale": 1,
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
    return train(config, prompts, device, on_step_complete, save_file=False)

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
    # --name 'eyesize_slider'
    parser.add_argument(
        "--positive",
        type=str,
        nargs='+',
        required=False,
        default=None,
        help="positive term(s)",
    )
     # --name 'eyesize_slider'
    parser.add_argument(
        "--negative",
        type=str,
        required=False,
        default=None,
        help="negative term(s)",
    )
    parser.add_argument(
        "--clip_index",
        type=int,
        required=True,
        help="0 based clip index (sdxl has two)",
    )   
    args = parser.parse_args()

    main(args)
