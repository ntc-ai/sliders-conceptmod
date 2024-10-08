import argparse
import numpy as np
import os
import shutil
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file

class BinDataHandler():
    def __init__(self, data):
        self.data = data

    def get_tensor(self, key):
        return self.data[key]

def read_tensors(file_path):
    if file_path.endswith(".safetensors"):
        f = safe_open(file_path, framework="pt", device="cpu")
        return f, set(f.keys())
    if file_path.endswith(".bin"):
        data = torch.load(file_path, map_location=torch.device('cpu'))
        f = BinDataHandler(data)
        return f, set(data.keys())
    return None, None

def merge(tensor_map, tensor_map1, strength, t):
    for k in tensor_map1[1]:
        k2 = k
        if t == 'transformer':
            k2 = k2.replace("lora_unet-", "transformer.")
            k2 = k2.replace("_down", "_A")
            k2 = k2.replace("_up", "_B")
            k2 = k2.replace("-", ".")
        if t == 'CLIP':
            k2 = k2.replace("-", "_")
        if t == 'T5':
            k2 = k2.replace("-", "_")
            k2 = k2.replace("lora_te2_", "lora_te3_")
            #lora_te3_encoder_block_13_layer_0_SelfAttention_q.lora_up.weight
            #k2 = k2.replace("SelfAttention_o", "self_attn.o_proj")
            #k2 = k2.replace("SelfAttention_q", "self_attn.q_proj")
            #k2 = k2.replace("SelfAttention_v", "self_attn.v_proj")
            #k2 = k2.replace("SelfAttention_k", "self_attn.k_proj")
            #k2 = k2.replace("encoder_block_", "text_model_encoder_layers_")
            #k2 = k2.replace("_layer_0", "")
        print("BEFORE", k, k2, tensor_map1[0].get_tensor(k).abs().sum())

        if 'alpha' in k:
            #pass
            #tensor_map[k2]=strength*torch.ones([1])
            tensor_map[k2]=tensor_map1[0].get_tensor(k).clone()#args.transformers_strength*tensor_map1[0].get_tensor(k)
        elif 'dora_scale' in k:
            #pass
            #k2 = k.replace("lora_scale", "dora_scale")
            #tensor_map[k2]=args.transformers_strength*tensor_map1[0].get_tensor(k).clone()
            tensor_map[k2]=tensor_map1[0].get_tensor(k).clone()#args.transformers_strength*tensor_map1[0].get_tensor(k)
        elif '_up' in k or '_A' in k:
            tensor_map[k2]=tensor_map1[0].get_tensor(k).clone()
        elif '_down' in k or '_B' in k:
            tensor_map[k2]=strength*tensor_map1[0].get_tensor(k).clone()
        else:
            assert False, k + " not supported"


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Merge two safetensor model files.')
    parser.add_argument('unet_model', type=str, help='The unet model safetensor file')
    parser.add_argument('encoder1_model', type=str, help='The encoder1 model safetensor file')
    parser.add_argument('encoder2_model', type=str, help='The encoder2 model safetensor file')
    parser.add_argument('output_model', type=str, help='The output merged model safetensor file')
    parser.add_argument('unet_strength', type=float, default=0.8, help='Strength of the unet')
    parser.add_argument('enc_strength', type=float, default=1, help='Strength of the unet')
    parser.add_argument('enc2_strength', type=float, default=1, help='Strength of the unet')
    args = parser.parse_args()

    tensor_map = {}

    tensor_map1 = read_tensors(args.unet_model)
    tensor_map2 = read_tensors(args.encoder1_model)
    tensor_map3 = read_tensors(args.encoder2_model)

    merge(tensor_map, tensor_map1, args.unet_strength, 'transformer')
    merge(tensor_map, tensor_map2, args.enc_strength, 'CLIP')
    merge(tensor_map, tensor_map3, args.enc2_strength, 'T5')
    save_file(tensor_map, args.output_model)

if __name__ == '__main__':
    main()
