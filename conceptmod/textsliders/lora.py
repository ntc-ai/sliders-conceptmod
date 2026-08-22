# ref:
# - https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# - https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import os
import math
from typing import Optional, List, Type, Set, Literal

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file


UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
#     "Transformer2DModel",  # どうやらこっちの方らしい？ # attn1, 2
    "Attention"
]
UNET_TARGET_REPLACE_MODULE_CONV = [
    "ResnetBlock2D",
    "Downsample2D",
    "Upsample2D",
    "DownBlock2D",
    "UpBlock2D",
    
]  # locon, 3clier

LORA_PREFIX_UNET = "lora_unet"

DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

TRAINING_METHODS = Literal[
    "noxattn",  # train all layers except x-attns and time_embed layers
    "innoxattn",  # train all layers except self attention layers
    "selfattn",  # ESD-u, train only self attention layers
    "xattn",  # ESD-x, train only x attention layers
    "full",  #  train all layers
    "xattn-strict", # q and k values
    "noxattn-hspace",
    "noxattn-hspace-last",
    # "xlayer",
    # "outxattn",
    # "outsattn",
    # "inxattn",
    # "inmidsattn",
    # "selflayer",
]




def lora_init_weights_advanced(lora_down, lora_up, init_scale=0.01, sparsity=0.9):
    fan_in = lora_down.weight.shape[1]
    fan_out = lora_up.weight.shape[0]
    mid_dim = lora_down.weight.shape[0]

    # Initialize lora_down with an asymmetric pattern
    with torch.no_grad():
        lora_down.weight.zero_()
        for i in range(mid_dim):
            if i % 2 == 0:
                lora_down.weight[i, i % fan_in] = init_scale
            else:
                lora_down.weight[i, -(i % fan_in) - 1] = -init_scale

    # Initialize lora_up with a complementary asymmetric pattern
    with torch.no_grad():
        lora_up.weight.zero_()
        for i in range(fan_out):
            if i % 2 == 0:
                lora_up.weight[i, i % mid_dim] = init_scale
            else:
                lora_up.weight[i, -(i % mid_dim) - 1] = -init_scale


class LoRAModule(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=None,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        elif org_module.__class__.__name__ == "Conv1d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
            self.lora_down = nn.Conv1d(
                in_dim,
                self.lora_dim,
                org_module.kernel_size,
                org_module.stride,
                org_module.padding,
                bias=False,
            )
            self.lora_up = nn.Conv1d(self.lora_dim, out_dim, 1, bias=False)

        elif "Conv" in org_module.__class__.__name__:  # 一応
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        #lora_init_weights_advanced(self.lora_down, self.lora_up, 5e-2, sparsity=0.9)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.seq_gain = None  # optional 1-D envelope over the sequence axis
        self.seq_gain_mode = "stretch"  # stretch | prefix
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _seq_gain_broadcast(self, x):
        gain = self.seq_gain
        if gain is None:
            return 1.0
        if x.dim() == 3:
            seq_dim, seq_len = 1, x.shape[1]
        elif x.dim() == 2:
            seq_dim, seq_len = 0, x.shape[0]
        else:
            return 1.0
        g = gain.to(device=x.device, dtype=x.dtype).reshape(-1)
        if g.numel() == 1:
            return g
        mode = getattr(self, "seq_gain_mode", "stretch")
        if mode == "prefix":
            if g.numel() >= seq_len:
                g = g[:seq_len]
            else:
                g = torch.cat((g, g[-1].expand(seq_len - g.numel())))
        elif g.numel() != seq_len:
            g = torch.nn.functional.interpolate(
                g.float().view(1, 1, -1),
                size=seq_len,
                mode="linear",
                align_corners=True,
            ).to(dtype=x.dtype).view(-1)
        shape = [1] * x.dim()
        shape[seq_dim] = seq_len
        return g.reshape(shape)

    def forward(self, x):
        # LoRA may be fp32/cuda while the host module is bf16 or CPU-offloaded.
        weight = self.lora_down.weight
        x_lora = x.to(device=weight.device, dtype=weight.dtype)
        delta = self.lora_up(self.lora_down(x_lora)).to(device=x.device, dtype=x.dtype)
        gain = self.multiplier * self.scale * self._seq_gain_broadcast(x)
        return self.org_forward(x) + delta * gain


class LoRANetwork(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        rank: int = 4,
        multiplier: float = 1.0,
        delimiter: str = "_",
        alpha: float = 1.0,
        target_replace = DEFAULT_TARGET_REPLACE,
        prefix=LORA_PREFIX_UNET,
        train_method: TRAINING_METHODS = "full",
    ) -> None:
        super().__init__()
        self.lora_scale = 1
        self.seq_gain = None
        self.seq_gain_mode = "stretch"
        self.multiplier = multiplier
        self.lora_dim = rank
        self.alpha = alpha

        # LoRAのみ
        self.module = LoRAModule

        # unetのloraを作る
        self.unet_loras = self.create_modules(
            prefix,
            unet,
            target_replace,
            delimiter,
            self.lora_dim,
            self.multiplier,
            train_method=train_method,
        )
        #print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        # assertion 名前の被りがないか確認しているようだ
        lora_names = set()
        for lora in self.unet_loras:
            assert (
                lora.lora_name not in lora_names
            ), f"duplicated lora name: {lora.lora_name}. {lora_names}"
            lora_names.add(lora.lora_name)

        # 適用する
        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(
                lora.lora_name,
                lora,
            )

        del unet

        torch.cuda.empty_cache()

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        delimiter: str,
        rank: int,
        multiplier: float,
        train_method: TRAINING_METHODS,
    ) -> list:
        loras = []
        names = []
        wrapped_ids = set()  # dedupe by module identity: overlapping target classes
        # (e.g. the root model + its blocks) must not wrap the same Linear twice
        for name, module in root_module.named_modules():
            if train_method == "noxattn" or train_method == "noxattn-hspace" or train_method == "noxattn-hspace-last":  # Cross Attention と Time Embed 以外学習
                if "attn2" in name or "time_embed" in name:
                    continue
            elif train_method == "innoxattn":  # Cross Attention 以外学習
                if "attn2" in name:
                    continue
            elif train_method == "selfattn":  # Self Attention のみ学習
                if "attn1" not in name:
                    continue
            elif train_method == "xattn" or train_method == "xattn-strict":  # Cross Attention のみ学習
                if "attn2" not in name:
                    continue
            elif train_method == "full":  # 全部学習
                pass
            else:
                raise NotImplementedError(
                    f"train_method: {train_method} is not implemented."
                )
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if 'add_' in child_name:
                        continue
                    if child_module.__class__.__name__ in [
                        "Linear",
                        "Conv1d",
                        "Conv2d",
                        "LoRACompatibleLinear",
                        "LoRACompatibleConv",
                    ]:
                        if train_method == 'xattn-strict':
                            if 'out' in child_name:
                                continue
                        if train_method == 'noxattn-hspace':
                            if 'mid_block' not in name:
                                continue
                        if train_method == 'noxattn-hspace-last':
                            if 'mid_block' not in name or '.1' not in name or 'conv2' not in child_name:
                                continue
                        if id(child_module) in wrapped_ids:
                            continue
                        # Skip empty path segments. The root model class is one of
                        # TARGET_REPLACE_FULL and its named_modules() name is "",
                        # which used to yield "lora_unet..transformer_blocks-0-..."
                        # -> a "lora_unet--" double delimiter. Because the identity
                        # dedupe lets the root claim every module first, *every*
                        # --targets full key came out double, disagreeing with the
                        # single-delimiter names --targets attn gives the very same
                        # attention modules -- and with the ComfyUI converter, which
                        # matches "lora_unet-".
                        lora_name = ".".join(p for p in (prefix, name, child_name) if p)
                        lora_name = lora_name.replace(".", delimiter)
#                         print(f"{lora_name}")
                        lora = self.module(
                            lora_name, child_module, multiplier, rank, self.alpha
                        )
#                         print(name, child_name)
#                         print(child_module.weight.shape)
                        if lora_name not in names:
                            wrapped_ids.add(id(child_module))
                            loras.append(lora)
                            names.append(lora_name)
#         print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n {names}')
        return loras

    def prepare_optimizer_params(self):
        all_params = []

        if self.unet_loras:  # 実質これしかない
            params = []
            [params.extend(lora.parameters()) for lora in self.unet_loras]
            param_data = {"params": params}
            all_params.append(param_data)

        return all_params

    def get_state_dict(self, dtype=None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v
        return state_dict

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

#         for key in list(state_dict.keys()):
#             if not key.startswith("lora"):
#                 # lora以外除外
#                 del state_dict[key]

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
    def set_lora_slider(self, scale):
        self.lora_scale = scale

    def set_seq_gain(self, gain, mode: str = "stretch"):
        """Per-position envelope for the LoRA delta. None = uniform.

        mode='stretch' interpolates the curve onto the current sequence
        (transformer chunks). mode='prefix' takes the first S values
        (LM tokens, including a full-prefix refresh).
        """
        if gain is None:
            tensor = None
        elif torch.is_tensor(gain):
            tensor = gain.detach().float().reshape(-1).cpu()
        else:
            tensor = torch.tensor(list(gain), dtype=torch.float32)
        self.seq_gain = tensor
        self.seq_gain_mode = mode
        for lora in self.unet_loras:
            lora.seq_gain = tensor
            lora.seq_gain_mode = mode

    def __enter__(self):
        for lora in self.unet_loras:
            lora.multiplier = 1.0 * self.lora_scale
            lora.seq_gain = self.seq_gain
            lora.seq_gain_mode = getattr(self, "seq_gain_mode", "stretch")

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras:
            lora.multiplier = 0
            lora.seq_gain = None
