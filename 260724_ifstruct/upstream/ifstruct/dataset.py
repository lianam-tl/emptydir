from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class IfStructExample:
    seed: int
    entity_type: str
    prompt: str
    json_schema: dict[str, Any]
    top_level_count: int | list[int]
    top_level_key: str | None
    require_wrapper_key: bool
    require_code_block: bool
    require_no_commentary: bool
    output_format: str


def load_examples(path: str | Path) -> list[IfStructExample]:
    examples: list[IfStructExample] = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            examples.append(IfStructExample(**row))
    return examples

