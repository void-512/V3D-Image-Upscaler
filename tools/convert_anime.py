#!/usr/bin/env python3
"""Convert the exact APISR DAT-Small checkpoint to the dedicated Vulkan format."""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import torch
from reference.dat import DAT, Spatial_Attention


def load_model(checkpoint):
    torch.set_num_threads(4)
    model = DAT(upscale=4, in_chans=3, img_size=64, img_range=1.,
                depth=[6] * 6, embed_dim=180, num_heads=[6] * 6,
                expansion_factor=2, resi_connection='1conv',
                split_size=[8, 16], upsampler='pixelshuffledirect').eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


@torch.inference_mode()
def convert(checkpoint, destination):
    model = load_model(checkpoint)
    tensors = {}
    G = torch.tensor([[.25, 0., 0.], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6], [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0., 0., 1.]])
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if '.pos' in name or '.attns.' in name:
            continue
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            weight = module.weight.detach().clone()
            bias = module.bias.detach().clone() if module.bias is not None else torch.zeros(weight.shape[0])
            # Evaluation BatchNorm is exactly affine: fold it into the preceding convolution.
            bn_name = None
            if name.endswith('.dwconv.0'):
                bn_name = name[:-1] + '1'
            elif name.endswith('.channel_interaction.1'):
                bn_name = name[:-1] + '2'
            elif name.endswith('.spatial_interaction.0'):
                bn_name = name[:-1] + '1'
            if bn_name:
                bn = modules[bn_name]
                scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                weight *= scale.reshape([-1] + [1] * (weight.ndim - 1))
                bias = (bias - bn.running_mean) * scale + bn.bias
            if weight.ndim == 4 and weight.shape[1:] == (180, 3, 3):
                transformed = (G @ weight @ G.t()).permute(2, 3, 1, 0).contiguous()
                tensors[name + '.winograd'] = transformed.reshape(36, 180, weight.shape[0] // 4, 4).permute(0, 2, 1, 3).contiguous()
                tensors[name + '.bias'] = bias
                continue
            if weight.ndim == 4:
                # K-major, output channels contiguous; depthwise uses [kh,kw,c].
                weight = weight.permute(2, 3, 1, 0).contiguous()
            else:
                weight = weight.t().contiguous()
                if name.endswith(('.attn.proj', '.ffn.fc1', '.ffn.fc2')):
                    assert weight.shape[0] == 180 and weight.shape[1] in (180, 360)
                    weight = weight.reshape(180, weight.shape[1] // 4, 4).permute(1, 0, 2).contiguous()
            tensors[name + '.weight'] = weight
            tensors[name + '.bias'] = bias
        elif isinstance(module, torch.nn.LayerNorm):
            tensors[name + '.weight'] = module.weight
            tensors[name + '.bias'] = module.bias
        if hasattr(module, 'temperature'):
            tensors[name + '.temperature'] = module.temperature
    for name, module in modules.items():
        if isinstance(module, Spatial_Attention):
            pos = module.pos(module.rpe_biases)
            bias = pos[module.relative_position_index.flatten()].reshape(128, 128, 3)
            tensors[name + '.relative_bias'] = bias.permute(2, 0, 1).contiguous()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries, arrays, offset = [], [], 0
    for name, value in tensors.items():
        arr = value.detach().cpu().float().numpy().reshape(-1).astype('<f4')
        if offset % 4:
            padding = 4 - offset % 4
            arrays.append(np.zeros(padding, dtype='<f4'))
            offset += padding
        entries.append((name, offset, arr.size))
        arrays.append(arr)
        offset += arr.size
    with destination.open('wb') as f:
        f.write(b'APISRDAT')
        f.write(struct.pack('<III', 2, len(entries), offset))
        for name, start, count in entries:
            encoded = name.encode()
            f.write(struct.pack('<III', len(encoded), start, count))
            f.write(encoded)
        for arr in arrays:
            f.write(arr.tobytes())
    metadata = {'format': 2, 'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                'model_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                'floats': offset, 'tensors': len(entries), 'precision': 'float32',
                'transforms': ['evaluation BatchNorm folding', 'constant relative-position bias evaluation',
                               'NHWC convolution and transposed linear weights',
                               '16-byte-aligned tensor starts', 'FP32 F(4x4,3x3) Winograd convolutions',
                               'quad-contiguous FP32 projection/FFN and Winograd weights']}
    destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(ROOT / 'models/checkpoints/4x_APISR_DAT_GAN_generator.pth'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    convert(args.checkpoint, args.output)
