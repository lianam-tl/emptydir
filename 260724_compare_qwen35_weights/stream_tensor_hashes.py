#!/usr/bin/env python3
"""Read one safetensors file from stdin and hash every tensor payload."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import sys


def read_exact(stream, byte_count: int) -> bytes:
    chunks = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(
                f"Expected {byte_count} bytes, received {byte_count - remaining}"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def hash_stream(
    stream,
    source: str,
    chunk_bytes: int = 16 * 1024 * 1024,
    inline_max_bytes: int = 0,
) -> dict:
    header_length = struct.unpack("<Q", read_exact(stream, 8))[0]
    header = json.loads(read_exact(stream, header_length))
    ordered_tensors = sorted(
        (
            (name, metadata)
            for name, metadata in header.items()
            if name != "__metadata__"
        ),
        key=lambda item: item[1]["data_offsets"][0],
    )

    cursor = 0
    tensors = {}
    for name, metadata in ordered_tensors:
        start, end = metadata["data_offsets"]
        if start < cursor:
            raise ValueError(f"Overlapping tensor payload for {name}")
        if start > cursor:
            read_exact(stream, start - cursor)
        digest = hashlib.sha256()
        remaining = end - start
        inline_chunks = [] if remaining <= inline_max_bytes else None
        while remaining:
            chunk = stream.read(min(chunk_bytes, remaining))
            if not chunk:
                raise EOFError(f"Unexpected EOF in tensor {name}")
            digest.update(chunk)
            if inline_chunks is not None:
                inline_chunks.append(chunk)
            remaining -= len(chunk)
        tensor = {
            "dtype": metadata["dtype"],
            "shape": metadata["shape"],
            "byte_count": end - start,
            "sha256": digest.hexdigest(),
        }
        if inline_chunks is not None:
            tensor["data_base64"] = base64.b64encode(b"".join(inline_chunks)).decode()
        tensors[name] = tensor
        cursor = end

    return {
        "source": source,
        "tensor_count": len(tensors),
        "payload_byte_count": cursor,
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inline-max-bytes", type=int, default=0)
    arguments = parser.parse_args()
    manifest = hash_stream(
        sys.stdin.buffer,
        arguments.source,
        inline_max_bytes=arguments.inline_max_bytes,
    )
    with open(arguments.output, "w") as output_file:
        json.dump(manifest, output_file, indent=2)
        output_file.write("\n")
    print(
        f"Hashed {manifest['tensor_count']} tensors "
        f"({manifest['payload_byte_count'] / 1024**3:.3f} GiB) from {arguments.source}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
