"""Pack model tensors into the embedded FP32/FP16 weight layout."""
from pathlib import Path
import struct

import numpy as np


MAGIC = {"anime": b"APISRDAT", "live": b"RRDBV3D!"}


def packed_weight(name, mode):
    if mode == "live":
        return name.endswith(".winograd")
    return name.endswith((".attn.qkv.weight", ".attn.proj.weight",
                          ".ffn.fc1.weight", ".ffn.fc2.weight"))


def write_model(mode, tensors, output):
    entries = []
    lengths = [0, 0]
    for name, tensor in sorted(tensors.items()):
        kind = int(packed_weight(name, mode))
        dtype = "<f2" if kind else "<f4"
        with np.errstate(over="raise", invalid="raise"):
            values = tensor.detach().cpu().float().numpy().reshape(-1).astype(dtype, copy=False)
        if kind and values.size % 4:
            raise ValueError(f"Unaligned packed tensor {name}")
        entries.append((name.encode(), values, kind, lengths[kind]))
        lengths[kind] += (values.size + 3) // 4 * 4

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(MAGIC[mode] + struct.pack("<IIII", 3, len(entries), *lengths))
            for name, values, kind, offset in entries:
                stream.write(struct.pack("<IIII", len(name), offset, values.size, kind))
                stream.write(name)
            for storage in range(2):
                for _, values, kind, _ in entries:
                    if storage == kind:
                        stream.write(values.tobytes())
                        stream.write(b"\0" * ((-values.size % 4) * values.itemsize))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
