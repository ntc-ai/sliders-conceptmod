from typing import Optional, Union

import torch

from math import ceil
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin, FluxPipeline, FluxTransformer2DModel
from diffusers.models.transformers import SD3Transformer2DModel
#from diffusers.schedulers import DDPMWuerstchenScheduler
from diffusers.utils.torch_utils import randn_tensor

from conceptmod.textsliders.model_util import SDXL_TEXT_ENCODER_TYPE

from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift

from tqdm import tqdm
import numpy as np

UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定。XLも同じ。
VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816


def get_random_noise(
    batch_size: int, height: int, width: int, generator: torch.Generator = None
) -> torch.Tensor:
    return torch.randn(
        (
            batch_size,
            UNET_IN_CHANNELS,
            height // VAE_SCALE_FACTOR,  # 縦と横これであってるのかわからないけど、どっちにしろ大きな問題は発生しないのでこれでいいや
            width // VAE_SCALE_FACTOR,
        ),
        generator=generator,
        device="cpu",
    )


# https://www.crosslabs.org/blog/diffusion-with-offset-noise
def apply_noise_offset(latents: torch.FloatTensor, noise_offset: float):
    latents = latents + noise_offset * torch.randn(
        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
    )
    return latents


def get_initial_latents_sd3(
    scheduler: SchedulerMixin,
    n_imgs: int,
    height: int,
    width: int,
    n_prompts: int,
    generator=None,
) -> torch.Tensor:
    return torch.randn(
        (
            n_imgs,
            16,
            height // VAE_SCALE_FACTOR,  # 縦と横これであってるのかわからないけど、どっちにしろ大きな問題は発生しないのでこれでいいや
            width // VAE_SCALE_FACTOR,
        ),
        device="cuda",
    )


def get_initial_latents_flux(
    scheduler: SchedulerMixin,
    n_imgs: int,
    height: int,
    width: int,
    n_prompts: int,
    generator=None,
) -> torch.Tensor:
    shape = (
            n_imgs,
            16,
            height // VAE_SCALE_FACTOR,
            width // VAE_SCALE_FACTOR,
        )
    #print("--!!", shape)
    return torch.randn(
        shape,
        device="cuda",
    )



def get_initial_latents(
    scheduler: SchedulerMixin,
    n_imgs: int,
    height: int,
    width: int,
    n_prompts: int,
    generator=None,
) -> torch.Tensor:
    noise = get_random_noise(n_imgs, height, width, generator=generator).repeat(
        n_prompts, 1, 1, 1
    )

    latents = noise * scheduler.init_noise_sigma.to(noise.device)

    return latents


def text_tokenize(
    tokenizer: CLIPTokenizer,  # 普通ならひとつ、XLならふたつ！
    prompts: list[str],
):
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids


def text_encode(text_encoder: CLIPTextModel, tokens):
    return text_encoder(tokens.to(text_encoder.device))[0]


def encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTokenizer,
    prompts: list[str],
):

    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)
    
    

    return text_embeddings


# https://github.com/huggingface/diffusers/blob/78922ed7c7e66c20aa95159c7b7a6057ba7d590d/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L334-L348
def text_encode_xl(
    text_encoder: SDXL_TEXT_ENCODER_TYPE,
    tokens: torch.FloatTensor,
    num_images_per_prompt: int = 1,
):
    prompt_embeds = text_encoder(
        tokens.to(text_encoder.device), output_hidden_states=True
    )
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]  # always penultimate layer

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds

def encode_prompts_cascade(
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTokenizer,
        prompts: list[str],
        ):

    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)



    return text_embeddings


def encode_prompts_sd3(
    pipeline,
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompt: str,
    num_images_per_prompt: int = 1,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    np = None
    (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=np,
            negative_prompt_2=np,
            negative_prompt_3=np,
            device="cuda",
            num_images_per_prompt=num_images_per_prompt,
        )

    return prompt_embeds, pooled_prompt_embeds

def encode_prompts_flux(
    pipeline,
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompt: str,
    num_images_per_prompt: int = 1,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    max_sequence_length = 512 #TODO
    (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
    ) = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device="cuda",
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    return prompt_embeds, pooled_prompt_embeds

def print_object_data(obj):
    data = {key: value for key, value in obj.__dict__.items() if not key.startswith('__')}
    for key, value in data.items():
        print(f"{key}")

def encode_prompts_xl(
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompts: list[str],
    num_images_per_prompt: int = 1,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    # text_encoder and text_encoder_2's penuultimate layer's output
    text_embeds_list = []
    pooled_text_embeds = None  # always text_encoder_2's pool

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_tokens_input_ids = text_tokenize(tokenizer, prompts)
        text_embeds, pooled_text_embeds = text_encode_xl(
            text_encoder, text_tokens_input_ids, num_images_per_prompt
        )

        text_embeds_list.append(text_embeds)

    bs_embed = pooled_text_embeds.shape[0]
    pooled_text_embeds = pooled_text_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )

    return torch.concat(text_embeds_list, dim=-1), pooled_text_embeds

def concat_embeddings_sd3(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    n_imgs: int,
):
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)


def concat_embeddings_flux(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    n_imgs: int,
):
    #print("OAHHH", unconditional.shape, conditional.shape)
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0).detach()



def concat_embeddings(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    n_imgs: int,
):
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)


# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L721
def predict_noise(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    guidance_scale=7.5,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    return guided_target


# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L746
@torch.no_grad()
def diffusion(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: torch.FloatTensor,
    total_timesteps: int = 1000,
    start_timesteps=0,
    **kwargs,
):
    # latents_steps = []

    for timestep in scheduler.timesteps[start_timesteps:total_timesteps]:
        noise_pred = predict_noise(
            unet, scheduler, timestep, latents, text_embeddings, **kwargs
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents


def rescale_noise_cfg(
    noise_cfg: torch.FloatTensor, noise_pred_text, guidance_rescale=0.0
):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )

    return noise_cfg

def predict_noise_sd3(
    transformer: SD3Transformer2DModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    guidance_scale=7.5
) -> torch.FloatTensor:

    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    timestep_cat = torch.tensor([timestep]*2*len(latents), device="cuda")
    #latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    # TODO timestep wrong 
    noise_pred = transformer(
        hidden_states=latent_model_input,
        timestep=timestep_cat,
        encoder_hidden_states=text_embeddings,
        pooled_projections=add_text_embeddings,
        return_dict=False,
    )[0]

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    latents = scheduler.step(guided_target, timestep, latents, return_dict=False)[0]

    return latents

def predict_noise_flux(
    pipeline: FluxPipeline,
    transformer: FluxTransformer2DModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    guidance_scale=7.5
) -> torch.FloatTensor:
    # https://github.com/huggingface/diffusers/blob/98930ee131b996c65cbbf48d8af363a98b21492c/src/diffusers/pipelines/flux/pipeline_flux.py#L508

    #TODO
    #latents = torch.cat([latents] * 2)
    generator = None
    height = pipeline.default_sample_size * pipeline.vae_scale_factor / 2
    width = pipeline.default_sample_size * pipeline.vae_scale_factor / 2
    num_channels_latents = transformer.config.in_channels // 4
    #TODO timestep
    device = text_embeddings.device
    #guidance = torch.tensor([guidance_scale], device=device)
    #guidance = guidance.expand(latents.shape[0])
    guidance = None
    batch_size = latents.shape[0]
    num_images_per_prompt = 1
    dtype = text_embeddings.dtype
    text_ids = torch.zeros(batch_size, text_embeddings.shape[1], 3).to(device=device, dtype=dtype)
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)
    #TODO move these to parent?

    height = 2 * (int(height) // pipeline.vae_scale_factor)
    width = 2 * (int(width) // pipeline.vae_scale_factor)
    latent_image_ids = pipeline._prepare_latent_image_ids(batch_size, height, width, device, dtype)

    timestep_ = timestep.expand(latents.shape[0]).to(latents.dtype)
    #print("--", latent_model_input.shape, timestep_.shape, 'guidance', guidance, add_text_embeddings.shape, text_embeddings.shape, text_ids.shape, latent_image_ids.shape)

    #_, add_text_embeddings = add_text_embeddings.chunk(2) #TODO
    #_, text_embeddings = text_embeddings.chunk(2)

    print('--ts', timestep_.mean())
    noise_pred = transformer(
        hidden_states=latents,
        timestep=timestep_ / 1000,
        guidance=guidance,
        pooled_projections=add_text_embeddings,
        encoder_hidden_states=text_embeddings,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]

    #TODO not needed?
    #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #guided_target = noise_pred_uncond + guidance_scale * (
    #    noise_pred_text - noise_pred_uncond
    #)
    #latents_2 = latent_model_input.chunk(2)[0]
    #print("call scheduler step", guided_target.shape, timestep.shape, latent_model_input.shape)
    #latents = scheduler.step(guided_target, timestep, latents, return_dict=False)[0]

    #print("call scheduler step", noise_pred.shape, timestep.shape, latent_model_input.shape)
    latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    return latents

def predict_noise_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    add_time_ids: torch.FloatTensor,
    guidance_scale=7.5,
    guidance_rescale=0.7,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    added_cond_kwargs = {
        "text_embeds": add_text_embeddings,
        "time_ids": add_time_ids,
    }

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    # https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L775
    noise_pred = rescale_noise_cfg(
        noise_pred_uncond, noise_pred_text, guidance_rescale=guidance_rescale
    )

    return guided_target


def predict_noise_cascade(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    prompt: str,
    batch_size=1,
    text_encoder=None,
    tokenizer=None,
    negative_prompt='',
    guidance_scale=7.5,
    guidance_rescale=0.7,
) -> torch.FloatTensor:
    latents = scheduler.scale_model_input(latents, timestep)
    dtype = next(unet.parameters()).dtype
    device = unet.device
    do_classifier_free_guidance=True
    num_images_per_prompt=1
    if hasattr(scheduler, "betas"):
        alphas = 1.0 - scheduler.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
    else:
        alphas_cumprod = []
    if not isinstance(scheduler, DDPMWuerstchenScheduler):
        if len(alphas_cumprod) > 0:
            ratio = get_t_condioning(timestep.long().cpu(), alphas_cumprod)
            ratio = ratio.expand(latents.size(0)).to(dtype).to(device)
        else:
            ratio = t.float().div(scheduler.timesteps[-1]).expand(latents.size(0)).to(dtype)
    else:
        ratio = t.expand(latents.size(0)).to(dtype)
    (
            prompt_embeds,
            prompt_embeds_pooled,
            negative_prompt_embeds,
            negative_prompt_embeds_pooled,
    ) = encode_prompt_cascade(
            prompt=prompt,
            device=device,
            batch_size=batch_size,#batch_size,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder
    )

    text_encoder_hidden_states = (
        torch.cat([prompt_embeds, negative_prompt_embeds]) if negative_prompt_embeds is not None else prompt_embeds
    )
    text_encoder_pooled = (
        torch.cat([prompt_embeds_pooled, negative_prompt_embeds_pooled])
        if negative_prompt_embeds is not None
        else prompt_embeds_pooled
    )

    image_embeds_pooled = torch.zeros(
        batch_size * num_images_per_prompt, 1, unet.config.c_clip_img, device=device, dtype=dtype
    )
    uncond_image_embeds_pooled = torch.zeros(
        batch_size * num_images_per_prompt, 1, unet.config.c_clip_img, device=device, dtype=dtype
    )
    if do_classifier_free_guidance:
        image_embeds = torch.cat([image_embeds_pooled, uncond_image_embeds_pooled], dim=0)
    else:
        image_embeds = image_embeds_pooled


    #print("LATENTS 1", latents.shape)
    # 7. Denoise image embeddings
    predicted_image_embedding = unet(
        x=torch.cat([latents] * 2) if do_classifier_free_guidance else latents,
        r=torch.cat([ratio] * 2) if do_classifier_free_guidance else ratio,
        clip_text_pooled=text_encoder_pooled,
        clip_text=text_encoder_hidden_states,
        clip_img=image_embeds
    )

    # 8. Check for classifier free guidance and apply it
    if do_classifier_free_guidance:
        predicted_image_embedding_text, predicted_image_embedding_uncond = predicted_image_embedding.chunk(2)
        predicted_image_embedding = torch.lerp(
            predicted_image_embedding_uncond, predicted_image_embedding_text, guidance_scale
        )
    return predicted_image_embedding

@torch.no_grad()
def diffusion_sd3(
    transformer,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: tuple[torch.FloatTensor, torch.FloatTensor],
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    guidance_scale: float = 1.0,
    total_timesteps: int = 1000,
    start_timesteps=0,
):
    # latents_steps = []

    for timestep in scheduler.timesteps[start_timesteps:total_timesteps]:
        latents = predict_noise_sd3(
            transformer,
            scheduler,
            timestep,
            latents,
            text_embeddings,
            add_text_embeddings,
            guidance_scale=guidance_scale
        )


    # return latents_steps
    return latents

@torch.no_grad()
def diffusion_flux(
    pipeline,
    transformer,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: tuple[torch.FloatTensor, torch.FloatTensor],
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    guidance_scale: float = 1.0,
    total_timesteps: int = 1000,
    start_timesteps=0,
):

    device = latents.device
    dtype = text_embeddings.dtype
    height = pipeline.default_sample_size * pipeline.vae_scale_factor / 2
    width = pipeline.default_sample_size * pipeline.vae_scale_factor / 2
    num_channels_latents = transformer.config.in_channels // 4
    batch_size = latents.shape[0]
    generator = None
    latents, latent_image_ids = pipeline.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents,
    )
    latents = latents.view(batch_size, -1, transformer.config.in_channels)

    #TODO?
    num_inference_steps = 1#total_timesteps+2# - start_timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.base_image_seq_len,
        pipeline.scheduler.config.max_image_seq_len,
        pipeline.scheduler.config.base_shift,
        pipeline.scheduler.config.max_shift,
    )
    timesteps = None
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )

    for timestep in timesteps[start_timesteps:total_timesteps]:
        latents = predict_noise_flux(
            pipeline,
            transformer,
            scheduler,
            timestep,
            latents,
            text_embeddings,
            add_text_embeddings,
            guidance_scale=guidance_scale
        )


    # return latents_steps
    return latents, timesteps



@torch.no_grad()
def diffusion_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: tuple[torch.FloatTensor, torch.FloatTensor],
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    add_time_ids: torch.FloatTensor,
    guidance_scale: float = 1.0,
    total_timesteps: int = 1000,
    start_timesteps=0,
):
    # latents_steps = []

    for timestep in scheduler.timesteps[start_timesteps:total_timesteps]:
        noise_pred = predict_noise_xl(
            unet,
            scheduler,
            timestep,
            latents,
            text_embeddings,
            add_text_embeddings,
            add_time_ids,
            guidance_scale=guidance_scale,
            guidance_rescale=0.7, #TODO
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents

def get_t_condioning(t, alphas_cumprod):
    s = torch.tensor([0.003])
    clamp_range = [0, 1]
    min_var = torch.cos(s / (1 + s) * torch.pi * 0.5) ** 2
    var = alphas_cumprod[t]
    var = var.clamp(*clamp_range)
    s, min_var = s.to(var.device), min_var.to(var.device)
    ratio = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
    return ratio


def encode_prompt_cascade(
    device,
    batch_size,
    num_images_per_prompt,
    do_classifier_free_guidance,
    prompt=None,
    negative_prompt="",
    prompt_embeds: Optional[torch.FloatTensor] = None,
    prompt_embeds_pooled: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds_pooled: Optional[torch.FloatTensor] = None,
    text_encoder=None,
    tokenizer=None
):
    if prompt_embeds is None:
        # get prompt text embeddings
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : tokenizer.model_max_length]
            attention_mask = attention_mask[:, : tokenizer.model_max_length]

        text_encoder_output = text_encoder(
            text_input_ids.to(device), attention_mask=attention_mask.to(device), output_hidden_states=True
        )
        prompt_embeds = text_encoder_output.hidden_states[-1]
        if prompt_embeds_pooled is None:
            prompt_embeds_pooled = text_encoder_output.text_embeds.unsqueeze(1)

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    prompt_embeds_pooled = prompt_embeds_pooled.to(dtype=text_encoder.dtype, device=device)
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    prompt_embeds_pooled = prompt_embeds_pooled.repeat_interleave(num_images_per_prompt, dim=0)
    seq_len = prompt_embeds_pooled.shape[1]
    prompt_embeds_pooled = prompt_embeds_pooled.view(batch_size * num_images_per_prompt, seq_len, -1)

    if negative_prompt_embeds is None and do_classifier_free_guidance:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds_text_encoder_output = text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=uncond_input.attention_mask.to(device),
            output_hidden_states=True,
        )

        negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.hidden_states[-1]
        negative_prompt_embeds_pooled = negative_prompt_embeds_text_encoder_output.text_embeds.unsqueeze(1)

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        seq_len = negative_prompt_embeds_pooled.shape[1]
        negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.to(
            dtype=text_encoder.dtype, device=device
        )
        negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )
        # done duplicates

    #print("___ABC", negative_prompt_embeds_pooled.shape, prompt_embeds_pooled.shape)

    return prompt_embeds, prompt_embeds_pooled, negative_prompt_embeds, negative_prompt_embeds_pooled

def prepare_latents_cascade(shape, dtype, device, generator, latents, scheduler):
    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        if latents.shape != shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
        latents = latents.to(device)

    latents = latents * scheduler.init_noise_sigma
    return latents

@torch.no_grad()
def diffusion_cascade(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    prompt,
    tokenizer,
    text_encoder,
    batch_size,
    width=1024,
    height=1024,
    negative_prompt='',
    guidance_scale: float = 1.0,
    total_timesteps: int = 1000,
    start_timesteps=0,
):
    # latents_steps = []
    device = unet.device
    dtype = next(unet.parameters()).dtype
    do_classifier_free_guidance=True
    num_images_per_prompt=1

    image_embeds_pooled = torch.zeros(
        num_images_per_prompt, 1, unet.config.c_clip_img, device=device, dtype=dtype
    )
    uncond_image_embeds_pooled = torch.zeros(
        num_images_per_prompt, 1, unet.config.c_clip_img, device=device, dtype=dtype
    )
    if do_classifier_free_guidance:
        image_embeds = torch.cat([image_embeds_pooled, uncond_image_embeds_pooled], dim=0)
    else:
        image_embeds = image_embeds_pooled


    (
            prompt_embeds,
            prompt_embeds_pooled,
            negative_prompt_embeds,
            negative_prompt_embeds_pooled,
    ) = encode_prompt_cascade(
            prompt=prompt,
            device=device,
            batch_size=batch_size,#batch_size,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder
    )
    resolution_multiple = 42.67
    latent_height = ceil(height / resolution_multiple)
    latent_width = ceil(width / resolution_multiple)

    effnet_features_shape = (
            num_images_per_prompt * batch_size,
            unet.config.c_in,
            latent_height,
            latent_width,
    )
    generator = None
    latents = None
    latents = prepare_latents_cascade(effnet_features_shape, dtype, device, generator, latents, scheduler)
    for t in scheduler.timesteps[start_timesteps:total_timesteps]:
        latent_model_input = scheduler.scale_model_input(latents, t)
        if hasattr(scheduler, "betas"):
            alphas = 1.0 - scheduler.betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            alphas_cumprod = []
        if not isinstance(scheduler, DDPMWuerstchenScheduler):
            if len(alphas_cumprod) > 0:
                ratio = get_t_condioning(t.long().cpu(), alphas_cumprod)
                ratio = ratio.expand(latent_model_input.size(0)).to(dtype).to(device)
            else:
                ratio = t.float().div(scheduler.timesteps[-1]).expand(latents.size(0)).to(dtype)
        else:
            ratio = t.expand(latents.size(0)).to(dtype)
        s = [1, 77, -1]

        #negative_prompt_embeds = negative_prompt_embeds.view(*s)
        #negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.view([s[0], 1, s[2]])
        #prompt_embeds = prompt_embeds.view(*s)
        #prompt_embeds_pooled = prompt_embeds_pooled.view([s[0], 1, s[2]])
        #print(prompt_embeds.shape, negative_prompt_embeds.shape)
        text_encoder_hidden_states = (
            torch.cat([prompt_embeds, negative_prompt_embeds]) if negative_prompt_embeds is not None else prompt_embeds
        )
        #print("__",prompt_embeds_pooled.shape, negative_prompt_embeds_pooled.shape)
        text_encoder_pooled = (
            torch.cat([prompt_embeds_pooled, negative_prompt_embeds_pooled])
            if negative_prompt_embeds is not None
            else prompt_embeds_pooled
        )

        #print(text_encoder_pooled.shape, text_encoder_hidden_states.shape, image_embeds.shape)

        #image_embeds = image_embeds.repeat([batch_size,1,1])
        #print("S", text_encoder_hidden_states.shape, text_encoder_pooled.shape, image_embeds.shape) 
        # 7. Denoise image embeddings
        #print("LATENTS 1", latent_model_input.shape)
        #print("RATIO 1", latent_model_input.shape)
        predicted_image_embedding = unet(
            x=torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input,
            r=torch.cat([ratio] * 2) if do_classifier_free_guidance else ratio,
            clip_text_pooled=text_encoder_pooled,
            clip_text=text_encoder_hidden_states,
            clip_img=image_embeds
        )

        # 8. Check for classifier free guidance and apply it
        if do_classifier_free_guidance:
            predicted_image_embedding_text, predicted_image_embedding_uncond = predicted_image_embedding.chunk(2)
            predicted_image_embedding = torch.lerp(
                predicted_image_embedding_uncond, predicted_image_embedding_text, guidance_scale
            )

        # 9. Renoise latents to next timestep
        if not isinstance(scheduler, DDPMWuerstchenScheduler):
            ratio = t
        latents = scheduler.step(
            model_output=predicted_image_embedding,
            timestep=ratio,
            sample=latents,
            generator=generator,
        ).prev_sample


    # return latents_steps
    return latents


# for XL
def get_add_time_ids(
    height: int,
    width: int,
    dynamic_crops: bool = False,
    dtype: torch.dtype = torch.float32,
):
    if dynamic_crops:
        # random float scale between 1 and 3
        random_scale = torch.rand(1).item() * 2 + 1
        original_size = (int(height * random_scale), int(width * random_scale))
        # random position
        crops_coords_top_left = (
            torch.randint(0, original_size[0] - height, (1,)).item(),
            torch.randint(0, original_size[1] - width, (1,)).item(),
        )
        target_size = (height, width)
    else:
        original_size = (height, width)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)

    # this is expected as 6
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    # this is expected as 2816
    passed_add_embed_dim = (
        UNET_ATTENTION_TIME_EMBED_DIM * len(add_time_ids)  # 256 * 6
        + TEXT_ENCODER_2_PROJECTION_DIM  # + 1280
    )
    if passed_add_embed_dim != UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM:
        raise ValueError(
            f"Model expects an added time embedding vector of length {UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


def get_optimizer(name: str):
    name = name.lower()

    if name.startswith("dadapt"):
        import dadaptation

        if name == "dadaptadam":
            return dadaptation.DAdaptAdam
        elif name == "dadaptlion":
            return dadaptation.DAdaptLion
        else:
            raise ValueError("DAdapt optimizer must be dadaptadam or dadaptlion")

    elif name.endswith("8bit"):  # 検証してない
        import bitsandbytes as bnb

        if name == "adam8bit":
            return bnb.optim.Adam8bit
        elif name == "lion8bit":
            return bnb.optim.Lion8bit
        else:
            raise ValueError("8bit optimizer must be adam8bit or lion8bit")

    else:
        if name == "adam":
            return torch.optim.Adam
        elif name == "adamw":
            return torch.optim.AdamW
        elif name == "lion":
            from lion_pytorch import Lion

            return Lion
        elif name == "prodigy":
            import prodigyopt
            
            return prodigyopt.Prodigy
        else:
            raise ValueError("Optimizer must be adam, adamw, lion or Prodigy")


def get_lr_scheduler(
    name: Optional[str],
    optimizer: torch.optim.Optimizer,
    max_iterations: Optional[int],
    lr_min: Optional[float],
    **kwargs,
):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iterations, eta_min=lr_min, **kwargs
        )
    elif name == "cosine_with_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max_iterations // 10, T_mult=2, eta_min=lr_min, **kwargs
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max_iterations // 100, gamma=0.999, **kwargs
        )
    elif name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, **kwargs)
    elif name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, factor=0.5, total_iters=max_iterations // 100, **kwargs
        )
    else:
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2

    step = 64

    min_step = min_resolution // step
    max_step = max_resolution // step

    height = torch.randint(min_step, max_step, (1,)).item() * step
    width = torch.randint(min_step, max_step, (1,)).item() * step

    return height, width
