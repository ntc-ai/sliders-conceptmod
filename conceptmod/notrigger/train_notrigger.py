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
import torch.nn.functional as F

from conceptmod.textsliders.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from conceptmod.textsliders.dora import DoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from conceptmod.textsliders import train_util
from conceptmod.textsliders import model_util
from conceptmod.textsliders.prompt_util import (
    PromptEmbedsCache,
    PromptEmbedsPair,
    PromptSettings,
    PromptEmbedsXL,
)
from conceptmod.textsliders import debug_util
from conceptmod.textsliders import config_util
from conceptmod.textsliders.config_util import RootConfig

from transformers import get_linear_schedule_with_warmup

import wandb
import yaml

NUM_IMAGES_PER_PROMPT = 1


def flush():
    torch.cuda.empty_cache()
    gc.collect()




def fixed_distance_loss(trainable_positive_embs, pos_tex_embs, fixed_distance):
    # Calculate the vector from trainable_positive_embs to pos_tex_embs
    diff_vector = pos_tex_embs - trainable_positive_embs

    # Calculate the current distance
    current_distance = torch.norm(diff_vector, dim=-1, keepdim=True)

    # Calculate the direction unit vector
    direction = diff_vector / (current_distance+1e-8)

    # Clamp the distance to move
    clamped_distance = torch.clamp(fixed_distance.unsqueeze(-1), -current_distance, current_distance)

    # Create a target that's at the clamped distance
    target = trainable_positive_embs + direction * clamped_distance

    # Calculate loss
    loss = ((trainable_positive_embs - target) ** 2).mean()

    return loss



def train(
    config: RootConfig,
    prompts: list[PromptSettings],
    device,
    on_step_complete,
    save_file=True,
    clip_index=0,
    peft_type='lora',
    rank=4,
    model="SDXL",
    attributes=None,
    positive=None,
    negative=None
):
    metadata = {
        "config": config.json(),
    }
    save_path = Path(config.save.path)

    if config.logging.verbose:
        print(metadata)

    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    if model == "SDXL" or model == "PonyXL":
        (
            tokenizers,
            text_encoders,
            unet,
            noise_scheduler,
        ) = model_util.load_models_xl(
            config.pretrained_model.name_or_path,
            scheduler_name=config.train.noise_scheduler
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
            scheduler_name=config.train.noise_scheduler
        )
        del unet2
        flush()
    elif model == "FLUX.1":
        (
            tokenizers,
            text_encoders,
            transformer,
            noise_scheduler,
            pipeline
        ) = model_util.load_models_flux(
            config.pretrained_model.name_or_path,
            scheduler_name=config.train.noise_scheduler,
            weight_dtype=weight_dtype,
            load_transformer=False

        )
        (
            tokenizers2,
            text_encoders2,
            transformer2,
            noise_scheduler,
            pipeline2
        ) = model_util.load_models_flux(
            config.pretrained_model.name_or_path,
            scheduler_name=config.train.noise_scheduler,
            weight_dtype=weight_dtype,
            load_transformer=False
        )


    elif model == "SD3-Medium":
        (
            tokenizers,
            text_encoders,
            unet,
            noise_scheduler,
            pipeline
        ) = model_util.load_models_sd3(
            config.pretrained_model.name_or_path,
            scheduler_name=config.train.noise_scheduler,
            weight_dtype=weight_dtype
        )
        del pipeline
        del unet
        flush()
        (
            tokenizers2,
            text_encoders2,
            unet2,
            noise_scheduler,
            pipeline2
        ) = model_util.load_models_sd3(
            config.pretrained_model.name_or_path,
            scheduler_name=config.train.noise_scheduler,
            weight_dtype=weight_dtype
        )
        del pipeline2
        del unet2
        flush()


    for text_encoder in text_encoders+text_encoders2:
        text_encoder.to(device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    index = clip_index
    prefix = ["lora_te1","lora_te2"][index]
    target_replace = ["CLIPAttention","CLIPSdpaAttention"]
    if model == "FLUX.1":
        if clip_index == 0:
            target_replace = ["CLIPSdpaAttention"]
        if clip_index == 1:
            target_replace = ["T5Attention"]

    train_method = config.network.training_method
    if index == 2:
        train_method = "t5attn"
 
    peft_text_encoder = text_encoders[index]
    static_text_encoder = text_encoders2[index]
    if peft_type == 'dora':
        peft_class = DoRANetwork
    else:
        peft_class = LoRANetwork

    network = peft_class(
        peft_text_encoder,
        rank=rank,
        multiplier=1.0,
        delimiter="_",
        alpha=config.network.alpha,
        target_replace=target_replace,
        prefix=prefix,
        train_method=train_method
    ).to(device, dtype=weight_dtype)
    network.requires_grad_(True)

    warmup_steps = 100
    cosine_steps = 900
    #optimizer = torch.optim.AdamW(network.parameters(), lr=config.train.lr)
    optimizer = torch.optim.SGD(network.parameters(), lr=config.train.lr )


    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=warmup_steps + cosine_steps
    )

    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=config.train.eta_min)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    criteria = torch.nn.MSELoss()

    # debug
    if config.logging.verbose:
        debug_util.check_requires_grad(network)
        debug_util.check_training_mode(network)

    pbar = tqdm(range(config.train.iterations))

    print("INDEX", index)
    chosenlayer = -1
    last_loss = None

    if positive is not None:
        positive = ", ".join(positive)
    if positive == "" or positive is None:
        positive = None
    else:
        tokens = train_util.text_tokenize(tokenizers2[index], [positive])
        pos_tex_embs = static_text_encoder(
            tokens.to(static_text_encoder.device), output_hidden_states=True
        ).hidden_states[chosenlayer]

    neu_tokens = train_util.text_tokenize(tokenizers[index], ['']).to(peft_text_encoder.device)
    neutral_tex_embs = static_text_encoder(
            neu_tokens, output_hidden_states=True
        ).hidden_states[chosenlayer]
    if negative is not None:
        negative = ", ".join(negative)
    if negative == "" or negative is None:
        negative = None
    else:
        tokens = train_util.text_tokenize(tokenizers2[index], [negative])
        neg_tex_embs = static_text_encoder(
            tokens.to(static_text_encoder.device), output_hidden_states=True
        ).hidden_states[chosenlayer]

    static_attribute_embs = []
    attribute_tokens = []
    attributes = [] # TODO test if attributes help
    for attribute in attributes:
        tokens = train_util.text_tokenize(tokenizers2[index], [attribute])
        attribute_tex_embs = static_text_encoder(
            tokens.to(static_text_encoder.device), output_hidden_states=True
        ).hidden_states[chosenlayer]
        static_attribute_embs.append(attribute_tex_embs)
        attribute_tokens.append(train_util.text_tokenize(tokenizers[index], [attribute]).to(peft_text_encoder.device))

    λp = config.train.lambda_similarity # positive contrastive
    λn = config.train.lambda_similarity # positive contrastive
    if(len(attributes) > 0):
        λs = 5e-4 / len(attributes) # stabilize (attributes)
    else:
        λs = 0
    stabilize_every = 10 * len(attributes)
    ploss = torch.zeros([1], device=static_text_encoder.device)
    nloss = torch.zeros([1], device=static_text_encoder.device)

    distance1 = None
    distance2 = None
    split = 20

    for i in pbar:
        with network:
            stabilize_loss = torch.zeros([1], device=static_text_encoder.device)
            if positive is not None:
                for l in network.unet_loras:
                    l.multiplier = 1.0

                trainable_positive_embs = peft_text_encoder(
                    neu_tokens, output_hidden_states=True
                ).hidden_states[chosenlayer]
                if i == 0:
                    distance1 = torch.norm( pos_tex_embs - trainable_positive_embs.detach() ).mean() / split

                if negative is None:
                    ploss = ((pos_tex_embs - trainable_positive_embs) ** 2).mean()
                else:
                    ploss = fixed_distance_loss(trainable_positive_embs, pos_tex_embs, distance1).mean()
                #pregularization = -fixed_distance_loss(trainable_positive_embs, neg_tex_embs, ndistance1).mean()
                #pregularization = 1/(((trainable_positive_embs - neg_tex_embs) ** 2).mean()+1e-8)
                #v1 = trainable_positive_embs-neutral_tex_embs
                #v2 = neg_tex_embs-neutral_tex_embs

                if negative is not None:
                    v1 = trainable_positive_embs-neutral_tex_embs
                    v2 = neg_tex_embs-neutral_tex_embs
                    v1r = pos_tex_embs - neutral_tex_embs
                    pregularization = torch.abs((
                        torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).squeeze() - torch.nn.functional.cosine_similarity(v1r.unsqueeze(0), v2.unsqueeze(0)).squeeze()
                    ).mean())
                    #pregularization += 1/(((v1 - v2) ** 2).mean()+1e-8)
                    pregularization += torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).squeeze().mean()
                    pregularization += 1/(((trainable_positive_embs - neg_tex_embs) ** 2).mean()+1e-8)

                if(len(attributes) > 0 and i % stabilize_every == 0):
                    for attribute_token, attr_b in zip(attribute_tokens, static_attribute_embs):
                        attr_a = peft_text_encoder(
                                attribute_token, output_hidden_states=True
                                ).hidden_states[chosenlayer]
                        stabilize_loss += torch.norm(attr_a - attr_b, p=2).mean()


            if negative is not None:
                for l in network.unet_loras:
                    l.multiplier = -1.0

                trainable_negative_embs = peft_text_encoder(
                        neu_tokens, output_hidden_states=True
                        ).hidden_states[chosenlayer]

                if i == 0:
                    distance2 = torch.norm( neg_tex_embs - trainable_negative_embs.detach() ).mean() / split 
                if positive is None:
                    nloss = ((neg_tex_embs - trainable_negative_embs) ** 2).mean()
                else:
                    nloss = fixed_distance_loss(trainable_negative_embs, neg_tex_embs, distance2).mean()
                #nregularization = -fixed_distance_loss(trainable_negative_embs, pos_tex_embs, ndistance2).mean()
                #nloss = ((neg_tex_embs - trainable_negative_embs) ** 2).mean()

                if positive is not None:
                    v1 = trainable_negative_embs-neutral_tex_embs
                    v1r = neg_tex_embs-neutral_tex_embs
                    v2 = pos_tex_embs-neutral_tex_embs
                    nregularization = torch.abs((
                        torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).squeeze() - torch.nn.functional.cosine_similarity(v1r.unsqueeze(0), v2.unsqueeze(0)).squeeze()
                    ).mean())
                    nregularization += torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).squeeze().mean()
                    #nregularization += 1/(((v1 - v2) ** 2).mean()+1e-8)
                    nregularization += 1/(((trainable_negative_embs-pos_tex_embs) ** 2).mean()+1e-8)

                    #regularization = 1 / ((trainable_negative_embs - trainable_positive_embs) ** 2).mean()
                if(len(attributes) > 0 and i % stabilize_every == 0):
                    for attribute_token, attr_b in zip(attribute_tokens, static_attribute_embs):
                        attr_a = peft_text_encoder(
                                attribute_token, output_hidden_states=True
                                ).hidden_states[chosenlayer]
                        stabilize_loss += torch.norm(attr_a - attr_b, p=2).mean()


        if positive is None:
            loss = nloss
            similarity = None
            stabilize = λs * stabilize_loss
        elif negative is None:
            loss = ploss
            similarity = None
            stabilize = λs * stabilize_loss
        else:
            loss = ploss + nloss
            similarity = λp * pregularization + λn * nregularization
            stabilize = λs * stabilize_loss
        if negative is None:
            full_loss = ((trainable_positive_embs - pos_tex_embs)**2).mean()
        elif positive is None:
            full_loss = ((trainable_negative_embs - neg_tex_embs)**2).mean()
        else:
            full_loss = ((pos_tex_embs - trainable_positive_embs)**2).mean() + ((neg_tex_embs - trainable_negative_embs)**2).mean()
        if i % 800 == 0 and i > 1000:
            if last_loss is not None and last_loss == full_loss.item():
                print("loss stopped moving. exitting early.")
                break
            last_loss = full_loss.item()
            print("reconstruction:", last_loss, "similarity:", similarity , "stabilize:", stabilize)
        if negative is not None:
            full_nloss = torch.norm(neg_tex_embs - trainable_negative_embs).mean()
            nperc = full_nloss / (distance2*split)
        else:
            nperc = 0
            full_nloss = torch.tensor(0)
        if positive is not None:
            full_ploss = torch.norm(pos_tex_embs - trainable_positive_embs).mean()
            pperc = full_ploss / (distance1*split)
        else:
            pperc = 0
            full_ploss = torch.tensor(0)
        # 1000倍しないとずっと0.000...になってしまって見た目的に面白くない
        if config.logging.use_wandb:
            wandb.log(
                {"loss": loss, "iteration": i, "lr": lr_scheduler.get_last_lr()[0]}
            )

        #loss = loss + 1e-4 * sum(p.pow(2.0).sum() for p in network.parameters())
        if positive is None or negative is None:
            loss.backward()
            w_n = 0
            w_p = 0
            w_r = 0
            similarity = torch.tensor(0)
        else:
            balance_p = pperc
            balance_n = nperc
            diff = abs(balance_p - balance_n)
            scale_factor = 1 + 4 * (1 - torch.exp(-diff / 0.05))

            w_p = (balance_p * scale_factor) / (balance_p * scale_factor + balance_n * scale_factor)
            w_n = (balance_n * scale_factor) / (balance_p * scale_factor + balance_n * scale_factor)
            w_r = min(0.95, (nperc+pperc)/2)

            full_nloss = w_n * balance_n
            full_ploss = w_p * balance_p
            loss = (w_p * balance_p + w_n * balance_n)
            similarity = (1.0-w_r) * similarity
            #loss = (balance_p * ploss + (1.0-balance_p) * nloss)/2
            #loss += (balance_n * nloss + (1.0-balance_n) * ploss)/2
            #w_n = (1.-balance_p + balance_n)/2.0
            #w_p = (1.-balance_n + balance_p)/2.0
            #loss.backward()
            (loss+similarity).backward()

            #(loss+similarity).backward()
        pbar.set_description(f"w_n {w_n:0.2f} w_p {w_p:0.2f} w_r {w_r:0.2f} ndist: {full_nloss.item():.3e}({nperc*100:.1f}%) pdist: {full_ploss.item():.3e}({pperc*100:.1f}%) Curriculum: {loss.item()*1000:.3f} similarity: {similarity.item():.3e} stabilize: {stabilize.item()*1000:0.3f} lr {scheduler.get_last_lr()}")
        #torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.2)
        torch.nn.utils.clip_grad_value_(network.parameters(), clip_value=1.0)

        optimizer.step()
        if i < warmup_steps:
            scheduler.step()
        else:
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
    
    device = torch.device(f"cuda:{args.device}")
    train(config, [], device, on_step_complete=None, positive = args.positive, negative=args.negative, clip_index=args.clip_index, peft_type=args.peft_type, rank=args.rank, model=args.model)

def train_lora(target, positive, negative, unconditional, alpha=1.0, rank=4, device=0, name=None, attributes=None, batch_size=1, config_file='data/config-xl.yaml', resolution=512, steps=None, on_step_complete=None, clip_index=0, peft_type='lora', model="SDXL"):
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
    print("Found", config)
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

    device = torch.device(f"cuda:{device}")
    return train(config, [], device, on_step_complete, positive=[positive], negative=[negative], save_file=False, clip_index=clip_index, peft_type=peft_type, rank=rank, model=model, attributes=attributes)

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
        "--peft_type",
        type=str,
        required=False,
        default="dora",
        help="dora (default) or lora",
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
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="SDXL",
        help="model SDXL or SD3-Medium",
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
