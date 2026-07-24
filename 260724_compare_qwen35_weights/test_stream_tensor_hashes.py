import hashlib
import io
import json
import struct

from compare_tensor_hashes import float32_to_bfloat16_bits
from stream_tensor_hashes import hash_stream


def make_safetensors(tensors):
    offset = 0
    header = {}
    payload = bytearray()
    for name, raw_values in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(raw_values)],
            "data_offsets": [offset, offset + len(raw_values)],
        }
        payload.extend(raw_values)
        offset += len(raw_values)
    header_bytes = json.dumps(header).encode()
    return struct.pack("<Q", len(header_bytes)) + header_bytes + payload


def test_hashes_every_tensor_payload():
    model = make_safetensors({"first": b"abc", "second": b"defg"})
    manifest = hash_stream(io.BytesIO(model), "test", inline_max_bytes=4)

    assert manifest["tensor_count"] == 2
    assert manifest["payload_byte_count"] == 7
    assert manifest["tensors"]["first"]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert (
        manifest["tensors"]["second"]["sha256"] == hashlib.sha256(b"defg").hexdigest()
    )
    assert manifest["tensors"]["first"]["data_base64"] == "YWJj"


def test_float32_to_bfloat16_uses_round_to_nearest_even():
    assert float32_to_bfloat16_bits(1.0) == 0x3F80
    assert float32_to_bfloat16_bits(1.0078125) == 0x3F81
