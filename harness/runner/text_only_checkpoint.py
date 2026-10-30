"""Copy a single-file Gemma 4 checkpoint keeping only the text model (language_model + lm_head).

vLLM 0.19.1 serves gemma-4-12b-it-qat-w4a16-ct only as Gemma4ForCausalLM (the hosts' override), and
that loader skips vision/audio weights by name ("vision_tower.", "embed_vision.", ...) but not the
12B's `model.vision_embedder.*`, so it stops with KeyError: 'vision_embedder.patch_dense.weight'.
The text weights are copied byte for byte, so the served model is unchanged.

    python runner/text_only_checkpoint.py SRC_DIR DST_DIR
"""

import json
import shutil
import struct
import sys
from pathlib import Path

KEEP = ('model.language_model.', 'lm_head.')
CHUNK = 64 * 1024 * 1024


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    source = src / 'model.safetensors'
    with open(source, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    data_start = 8 + n
    kept = [(name, meta) for name, meta in header.items() if name != '__metadata__' and name.startswith(KEEP)]
    dropped = [name for name in header if name != '__metadata__' and not name.startswith(KEEP)]

    new_header = {'__metadata__': header.get('__metadata__', {'format': 'pt'})}
    offset = 0
    for name, meta in kept:
        length = meta['data_offsets'][1] - meta['data_offsets'][0]
        new_header[name] = {'dtype': meta['dtype'], 'shape': meta['shape'], 'data_offsets': [offset, offset + length]}
        offset += length
    blob = json.dumps(new_header, separators=(',', ':')).encode()
    blob += b' ' * (-len(blob) % 8)
    with open(source, 'rb') as f, open(dst / 'model.safetensors', 'wb') as out:
        out.write(struct.pack('<Q', len(blob)))
        out.write(blob)
        for _, meta in kept:
            f.seek(data_start + meta['data_offsets'][0])
            remaining = meta['data_offsets'][1] - meta['data_offsets'][0]
            while remaining:
                chunk = f.read(min(CHUNK, remaining))
                out.write(chunk)
                remaining -= len(chunk)
    for path in src.iterdir():
        if path.is_file() and path.name != 'model.safetensors':
            shutil.copy2(path, dst / path.name)
    (dst / 'TEXT_ONLY_DONE').write_text(f'kept {len(kept)} tensors, dropped {len(dropped)}: {dropped}\n')
    print(f'kept {len(kept)} tensors ({offset / 1e9:.2f} GB), dropped {len(dropped)}: {dropped}')


if __name__ == '__main__':
    main()
