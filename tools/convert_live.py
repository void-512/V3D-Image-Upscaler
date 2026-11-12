#!/usr/bin/env python3
"""Convert the exact RealESRGAN x4plus checkpoint to the dedicated Vulkan format."""
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
from reference.rrdbnet_arch import RRDBNet


def load_model(checkpoint):
    torch.set_num_threads(4)
    model = RRDBNet(3, 3, scale=4, num_feat=64, num_block=23, num_grow_ch=32).eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['params_ema']
    model.load_state_dict(state, strict=True)
    return model


@torch.inference_mode()
def convert(checkpoint, destination):
    model = load_model(checkpoint)
    tensors = {}
    G = torch.tensor([[.25, 0., 0.], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6], [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0., 0., 1.]])
    for name, value in model.state_dict().items():
        if value.ndim == 4:
            if name.startswith('body.') or name == 'conv_body.weight':
                transformed = G @ value @ G.t()
                transformed = transformed.permute(2, 3, 1, 0).contiguous()
                if value.shape[:2] == (64, 192):
                    transformed = transformed.reshape(36, 192, 16, 4).permute(0, 2, 1, 3).contiguous()
                tensors[name.replace('.weight', '.winograd')] = transformed
                continue
            value = value.permute(2, 3, 1, 0).contiguous()
            if name in ('conv_up1.weight', 'conv_up2.weight'):
                # Nearest-neighbour 2x expansion repeats input samples. Sum
                # their 3x3 coefficients into four parity-specific 2x2 kernels.
                phase = torch.zeros((2, 2, 2, 2, 64, 64), dtype=torch.float32)
                for py in range(2):
                    for px in range(2):
                        for ky in range(3):
                            for kx in range(3):
                                iy = (ky + py - 1) // 2 + 1 - py
                                ix = (kx + px - 1) // 2 + 1 - px
                                phase[py, px, iy, ix] += value[ky, kx]
                tensors[name.replace('.weight', '.polyphase')] = phase
        tensors[name] = value
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
        f.write(b'RRDBV3D!')
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
                'transforms': ['NHWC convolution weights', '16-byte-aligned tensor starts'] +
                              ['FP32 F(4x4,3x3) Winograd trunk weights', 'FP32 polyphase 2x upsampling kernels',
                               'quad-contiguous FP32 192-to-64 Winograd weights'],
                'checkpoint_key': 'params_ema', 'architecture': 'RRDBNet(3,3,64,23,32,scale=4)'}
    destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(ROOT / 'models/checkpoints/RealESRGAN_x4plus.pth'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    convert(args.checkpoint, args.output)
