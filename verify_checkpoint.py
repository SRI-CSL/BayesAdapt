from safetensors import safe_open
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, default='log/adapter_model.safetensors', help='Path to the .safetensors file')
args = parser.parse_args()

#path = 'checkpoints/scalabl/Qwen/Qwen2.5-7B/ARC-Easy/scalabl-ARC-Easy-sample10-eps0.05-kllr0.2-beta0.2-gamma8-seed1-id/adapter_model.safetensors'
# path = 'log/adapter_model.safetensors'
tensors = {}
with safe_open(args.path, framework="pt", device=1) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
        print(k, tensors[k].shape)

import ipdb; ipdb.set_trace() # noqa
