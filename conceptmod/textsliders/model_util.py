from typing import Literal, Union, Optional

import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
#from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
import os
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings

from diffusers import (
    UNet2DConditionModel,
    SchedulerMixin,
    StableDiffusionPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
    FluxPipeline,
    FluxTransformer2DModel
)
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)


TOKENIZER_V1_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
TOKENIZER_V2_MODEL_NAME = "stabilityai/stable-diffusion-2-1"

AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a"]

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]

DIFFUSERS_CACHE_DIR = None  # if you want to change the cache dir, change this


def load_diffusers_model(
    pretrained_model_name_or_path: str,
    v2: bool = False,
    clip_skip: Optional[int] = None,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel,]:
    # VAE はいらない

    if v2:
        tokenizer = CLIPTokenizer.from_pretrained(
            TOKENIZER_V2_MODEL_NAME,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            # default is clip skip 2
            num_hidden_layers=24 - (clip_skip - 1) if clip_skip is not None else 23,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            TOKENIZER_V1_MODEL_NAME,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            num_hidden_layers=12 - (clip_skip - 1) if clip_skip is not None else 12,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    return tokenizer, text_encoder, unet


def load_checkpoint_model(
    checkpoint_path: str,
    v2: bool = False,
    clip_skip: Optional[int] = None,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel,]:
    pipe = StableDiffusionPipeline.from_ckpt(
        checkpoint_path,
        upcast_attention=True if v2 else False,
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    if clip_skip is not None:
        if v2:
            text_encoder.config.num_hidden_layers = 24 - (clip_skip - 1)
        else:
            text_encoder.config.num_hidden_layers = 12 - (clip_skip - 1)

    del pipe

    return tokenizer, text_encoder, unet


def load_models(
    pretrained_model_name_or_path: str,
    scheduler_name: AVAILABLE_SCHEDULERS,
    v2: bool = False,
    v_pred: bool = False,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, SchedulerMixin,]:
    if pretrained_model_name_or_path.endswith(
        ".ckpt"
    ) or pretrained_model_name_or_path.endswith(".safetensors"):
        tokenizer, text_encoder, unet = load_checkpoint_model(
            pretrained_model_name_or_path, v2=v2, weight_dtype=weight_dtype
        )
    else:  # diffusers
        tokenizer, text_encoder, unet = load_diffusers_model(
            pretrained_model_name_or_path, v2=v2, weight_dtype=weight_dtype
        )

    # VAE はいらない

    scheduler = create_noise_scheduler(
        scheduler_name,
        prediction_type="v_prediction" if v_pred else "epsilon",
    )

    return tokenizer, text_encoder, unet, scheduler

def load_diffusers_model_cascade(
    pretrained_model_name_or_path: str,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[list[CLIPTokenizer], list[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel,]:
    # returns tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet

    #config = CLIPConfig.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    #config.text_config.projection_dim = config.projection_dim
    #text_encoder = CLIPTextModelWithProjection.from_pretrained(
    #    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", config=config.text_config
    #)
    #tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    prior_model = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to("cuda:0")

    print("___", sorted(dir(prior_model)))


    return prior_model.tokenizer, prior_model.text_encoder, prior_model.prior


def print_object_data(obj):
    data = {key: value for key, value in obj.__dict__.items() if not key.startswith('__')}
    for key, value in data.items():
        print(f"{key}")

def load_diffusers_model_xl(
    pretrained_model_name_or_path: str,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[list[CLIPTokenizer], list[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel,]:
    # returns tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet

    tokenizers = [
        CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
        CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
            pad_token_id=0,  # same as open clip
        ),
    ]

    text_encoders = [
        CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
        CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
    ]

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    return tokenizers, text_encoders, unet

def load_checkpoint_model_cascade(
    checkpoint_path: str,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[list[CLIPTokenizer], list[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel,]:
    assert False, "not implemented"


def load_checkpoint_model_sd3 (
    checkpoint_path: str,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[list[CLIPTokenizer], list[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel,]:
    #pipe = StableDiffusion3Pipeline.from_single_file(checkpoint_path, text_encoder_3=None, tokenizer_3=None, torch_dtype=torch.float16, use_safetensors=True, local_files_only=True, device="cuda")
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    #text_encoder = T5EncoderModel.from_pretrained(
    #    model_id,
    #    subfolder="text_encoder_3",
    #    torch_dtype = weight_dtype
    #)
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, use_safetensors=True, local_files_only=True, torch_device="cuda", torch_dtype=weight_dtype, text_encoder_3=None)
    pipe = pipe.to("cuda")

    transformer = pipe.transformer
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]

    return tokenizers, text_encoders, transformer, pipe

def load_checkpoint_model_flux (
    checkpoint_path: str,
    weight_dtype: torch.dtype = torch.float32,
    load_transformer=True,
) -> tuple[list[CLIPTokenizer], list[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel,]:
    #transformer = FluxTransformer2DModel.from_single_file(checkpoint_path,
    #    torch_dtype=weight_dtype,
    #    cache_dir=DIFFUSERS_CACHE_DIR,
    #    device="cuda",
    #    weights_only=False
    #        )
    #pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype)
    #pipe.transformer = transformer
    #pipe.text_encoder_2 = text_encoder_2
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", transformer=None, torch_dtype=torch.bfloat16)
    #pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", transformer=None, torch_dtype=torch.bfloat16)
    #pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

    if(load_transformer):
        transformer = FluxTransformer2DModel.from_single_file(checkpoint_path, torch_dtype=weight_dtype).cuda()
        pipe.transformer = transformer
    else:
        transformer = None
    #    
    #transformer.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
    #    embedding_dim=transformer.inner_dim, pooled_projection_dim=transformer.config.pooled_projection_dim
    #).to(torch.bfloat16)
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder.cuda(), pipe.text_encoder_2.cuda()]

    return tokenizers, text_encoders, pipe.transformer, pipe

def load_checkpoint_model_xl(
    checkpoint_path: str,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[list[CLIPTokenizer], list[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel,]:
    pipe = StableDiffusionXLPipeline.from_single_file(
        checkpoint_path,
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    unet = pipe.unet
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    if len(text_encoders) == 2:
        text_encoders[1].pad_token_id = 0

    del pipe

    return tokenizers, text_encoders, unet

def load_models_cascade(
    pretrained_model_name_or_path: str,
    scheduler_name: AVAILABLE_SCHEDULERS,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[
    list[CLIPTokenizer],
    list[SDXL_TEXT_ENCODER_TYPE],
    UNet2DConditionModel,
    SchedulerMixin,
]:
    (
        tokenizers,
        text_encoders,
        unet,
    ) = load_diffusers_model_cascade(pretrained_model_name_or_path, weight_dtype)

    scheduler = create_noise_scheduler(scheduler_name)

    return tokenizers, text_encoders, unet, scheduler



def load_models_sd3(
    pretrained_model_name_or_path: str,
    scheduler_name: AVAILABLE_SCHEDULERS,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[
    list[CLIPTokenizer],
    list[SDXL_TEXT_ENCODER_TYPE],
    UNet2DConditionModel,
    SchedulerMixin,
]:
    (
        tokenizers,
        text_encoders,
        unet,
        pipe
    ) = load_checkpoint_model_sd3(pretrained_model_name_or_path, weight_dtype)

    scheduler = pipe.scheduler#create_noise_scheduler(scheduler_name)

    return tokenizers, text_encoders, unet, scheduler, pipe

def load_models_flux(
    pretrained_model_name_or_path: str,
    scheduler_name: AVAILABLE_SCHEDULERS,
    load_transformer=True,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[
    list[CLIPTokenizer],
    list[SDXL_TEXT_ENCODER_TYPE],
    UNet2DConditionModel,
    SchedulerMixin,
]:
    (
        tokenizers,
        text_encoders,
        unet,
        pipe
    ) = load_checkpoint_model_flux(pretrained_model_name_or_path, weight_dtype, load_transformer=load_transformer)

    scheduler = pipe.scheduler#create_noise_scheduler(scheduler_name)
    print("scheduler", scheduler)

    return tokenizers, text_encoders, unet, scheduler, pipe



def load_models_xl(
    pretrained_model_name_or_path: str,
    scheduler_name: AVAILABLE_SCHEDULERS,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[
    list[CLIPTokenizer],
    list[SDXL_TEXT_ENCODER_TYPE],
    UNet2DConditionModel,
    SchedulerMixin,
]:
    if pretrained_model_name_or_path.endswith(
        ".ckpt"
    ) or pretrained_model_name_or_path.endswith(".safetensors"):
        (
            tokenizers,
            text_encoders,
            unet,
        ) = load_checkpoint_model_xl(pretrained_model_name_or_path, weight_dtype)
    else:  # diffusers
        (
            tokenizers,
            text_encoders,
            unet,
        ) = load_diffusers_model_xl(pretrained_model_name_or_path, weight_dtype)

    scheduler = create_noise_scheduler(scheduler_name)

    return tokenizers, text_encoders, unet, scheduler


def create_noise_scheduler(
    scheduler_name: AVAILABLE_SCHEDULERS = "ddpm",
    prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
) -> SchedulerMixin:
    # 正直、どれがいいのかわからない。元の実装だとDDIMとDDPMとLMSを選べたのだけど、どれがいいのかわからぬ。

    name = scheduler_name.lower().replace(" ", "_")
    if name == "ddim":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddim
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,  # これでいいの？
        )
    elif name == "ddpm":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddpm
        scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,
        )
    elif name == "lms":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/lms_discrete
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
    elif name == "euler_a":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/euler_ancestral
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
    else:
        raise ValueError(f"Unknown scheduler name: {name}")

    return scheduler
