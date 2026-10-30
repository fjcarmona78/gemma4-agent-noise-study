"""Split a safetensors checkpoint into smaller shards without memory-mapping the originals.

WSL refuses to mmap a single 49.8 GB shard (more than RAM + swap), so transformers cannot
load google/gemma-4-31B-it-qat-q4_0-unquantized as shipped. This copies every tensor with
plain seek/read into shards of at most --max-gb, and copies the other model files.

    python finetune/reshard.py SRC_DIR DST_DIR [--max-gb 5]
"""

import argparse
import json
import shutil
import struct
from pathlib import Path

CHUNK = 64 * 1024 * 1024


def read_header(path: Path) -> tuple[int, dict]:
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def write_shard(dst: Path, entries: list[tuple[str, dict, Path, int]]) -> int:
    """entries: (name, meta, source file, absolute byte offset of the tensor in that file)."""
    header = {'__metadata__': {'format': 'pt'}}
    offset = 0
    for name, meta, _, _ in entries:
        length = meta['data_offsets'][1] - meta['data_offsets'][0]
        header[name] = {'dtype': meta['dtype'], 'shape': meta['shape'],
                        'data_offsets': [offset, offset + length]}
        offset += length
    blob = json.dumps(header, separators=(',', ':')).encode()
    blob += b' ' * (-len(blob) % 8)
    with open(dst, 'wb') as out:
        out.write(struct.pack('<Q', len(blob)))
        out.write(blob)
        for _, meta, src, start in entries:
            remaining = meta['data_offsets'][1] - meta['data_offsets'][0]
            with open(src, 'rb') as f:
                f.seek(start)
                while remaining:
                    chunk = f.read(min(CHUNK, remaining))
                    out.write(chunk)
                    remaining -= len(chunk)
    return offset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=Path)
    parser.add_argument('dst', type=Path)
    parser.add_argument('--max-gb', type=float, default=5.0)
    args = parser.parse_args()
    max_bytes = int(args.max_gb * 1e9)
    args.dst.mkdir(parents=True, exist_ok=True)

    tensors = []
    for shard in sorted(args.src.glob('*.safetensors')):
        data_start, header = read_header(shard)
        for name, meta in header.items():
            if name != '__metadata__':
                tensors.append((name, meta, shard, data_start + meta['data_offsets'][0]))

    groups, current, size = [], [], 0
    for entry in tensors:
        length = entry[1]['data_offsets'][1] - entry[1]['data_offsets'][0]
        if current and size + length > max_bytes:
            groups.append(current)
            current, size = [], 0
        current.append(entry)
        size += length
    if current:
        groups.append(current)

    weight_map, total = {}, 0
    for i, group in enumerate(groups, 1):
        filename = f'model-{i:05d}-of-{len(groups):05d}.safetensors'
        total += write_shard(args.dst / filename, group)
        for name, *_ in group:
            weight_map[name] = filename
        print(f'wrote {filename} ({len(group)} tensors)', flush=True)

    index = {'metadata': {'total_size': total}, 'weight_map': weight_map}
    (args.dst / 'model.safetensors.index.json').write_text(json.dumps(index, indent=1))
    for path in args.src.iterdir():
        if path.is_file() and path.suffix != '.safetensors' and path.name != 'model.safetensors.index.json':
            shutil.copy2(path, args.dst / path.name)
    (args.dst / 'RESHARD_DONE').write_text(f'{len(tensors)} tensors in {len(groups)} shards\n')
    print(f'done: {len(tensors)} tensors, {len(groups)} shards, {total / 1e9:.1f} GB', flush=True)


if __name__ == '__main__':
    main()
