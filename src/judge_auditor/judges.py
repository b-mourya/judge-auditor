"""Judges: anything that looks at a question and two answers and picks the better one.

Every judge returns a *positional* verdict: "1" (first answer shown), "2" (second answer
shown) or "tie". The audits translate positions back into answer identities, which is
what lets us detect position bias.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import Pair

VERDICTS = ("1", "2", "tie")

JUDGE_PROMPT = """You are an impartial judge. Compare two answers to the user's question and decide which one is better. Judge factual correctness first, then helpfulness. Do not let the length of the answers or the order they appear in influence your decision.

[Question]
{question}

[Answer 1]
{first}

[Answer 2]
{second}

Explain your reasoning in a few sentences, then end with exactly one verdict on its own line:
[[1]] if Answer 1 is better, [[2]] if Answer 2 is better, or [[TIE]] if they are equally good."""

_VERDICT_RE = re.compile(r"\[\[\s*(1|2|TIE)\s*\]\]", re.IGNORECASE)


def parse_verdict(text: str) -> str:
    """Pull the final [[1]] / [[2]] / [[TIE]] verdict out of a judge's response."""
    matches = _VERDICT_RE.findall(text)
    if not matches:
        raise ValueError(f"No verdict found in judge output: {text[-200:]!r}")
    v = matches[-1].lower()
    return "tie" if v == "tie" else v


class Judge(ABC):
    name: str = "judge"

    @abstractmethod
    def compare(self, question: str, first: str, second: str) -> str:
        """Return "1", "2" or "tie"."""


# ---------------------------------------------------------------- real LLM judges


class AnthropicJudge(Judge):
    def __init__(self, model: str = "claude-haiku-5-5"):
        import anthropic  # imported lazily so the package works without it

        self.model = model
        self.name = f"anthropic:{model}"
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def compare(self, question: str, first: str, second: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=600,
            temperature=0,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, first=first, second=second)}],
        )
        return parse_verdict(resp.content[0].text)


class OpenAIJudge(Judge):
    def __init__(self, model: str):
        import openai

        self.model = model
        self.name = f"openai:{model}"
        self._client = openai.OpenAI()  # reads OPENAI_API_KEY

    def compare(self, question: str, first: str, second: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, first=first, second=second)}],
        )
        return parse_verdict(resp.choices[0].message.content or "")


# ---------------------------------------------------------------- simulated judge


class SimulatedJudge(Judge):
    """A fake judge with biases you can dial up or down.

    It "knows" answer quality from the human labels, then adds noise, a preference for
    whichever answer is shown first, and a preference for longer answers. Use it to run
    the pipeline for free and to check that each audit actually detects the bias it
    targets (that's what the tests do).
    """

    def __init__(
        self,
        pairs: list["Pair"],
        noise: float = 0.15,
        position_bias: float = 0.0,
        length_bias: float = 0.0,
        seed: int = 0,
    ):
        self.noise, self.position_bias, self.length_bias, self.seed = (
            noise, position_bias, length_bias, seed)
        self.name = f"simulated(noise={noise}, position_bias={position_bias}, length_bias={length_bias})"
        scores = {"A": (0.8, 0.2), "B": (0.2, 0.8), "tie": (0.5, 0.5)}
        self._quality: dict[tuple[str, str], float] = {}
        for p in pairs:
            qa, qb = scores[p.human]
            self._quality[(p.question, p.answer_a)] = qa
            self._quality[(p.question, p.answer_b)] = qb

    def _score(self, question: str, text: str) -> float:
        if (question, text) in self._quality:
            return self._quality[(question, text)]
        for (q, ans), s in self._quality.items():  # padded answers contain the original
            if q == question and ans in text:
                return s
        return 0.5

    def compare(self, question: str, first: str, second: str) -> str:
        key = f"{self.seed}|{question}|{first}|{second}".encode()
        rng = random.Random(hashlib.sha256(key).hexdigest())
        diff = (
            self._score(question, first) - self._score(question, second)
            + self.position_bias
            + self.length_bias * math.tanh((len(first) - len(second)) / 150)
            + rng.gauss(0, self.noise)
        )
        if diff > 0.2:
            return "1"
        if diff < -0.2:
            return "2"
        return "tie"


# ---------------------------------------------------------------- caching


class CachedJudge(Judge):
    """Wraps a judge and stores every verdict on disk, so re-running an audit is free."""

    def __init__(self, inner: Judge, cache_dir: str | os.PathLike = ".cache"):
        self.inner = inner
        self.name = inner.name
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", inner.name)
        self.path = Path(cache_dir) / f"{safe}.jsonl"
        self._cache: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row["verdict"]

    @staticmethod
    def _key(question: str, first: str, second: str) -> str:
        return hashlib.sha256(json.dumps([question, first, second]).encode()).hexdigest()

    def compare(self, question: str, first: str, second: str) -> str:
        k = self._key(question, first, second)
        if k not in self._cache:
            verdict = self.inner.compare(question, first, second)
            self._cache[k] = verdict
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "verdict": verdict}) + "\n")
        return self._cache[k]
