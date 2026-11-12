"""Loading the labeled comparison dataset."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HUMAN_LABELS = ("A", "B", "tie")


@dataclass(frozen=True)
class Pair:
    id: str
    question: str
    answer_a: str
    answer_b: str
    human: str  # "A", "B" or "tie": which answer a human judged better


def load_pairs(path: str | Path) -> list[Pair]:
    pairs = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("human") not in HUMAN_LABELS:
            raise ValueError(f"{path}:{n}: 'human' must be one of {HUMAN_LABELS}")
        pairs.append(Pair(
            id=str(row["id"]), question=row["question"],
            answer_a=row["answer_a"], answer_b=row["answer_b"], human=row["human"],
        ))
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return pairs
