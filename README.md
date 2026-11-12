# judge-auditor

**How much should you trust an LLM-as-judge?**

Teams increasingly use one LLM to grade another LLM's answers. It's fast and cheap, but LLM judges have known failure modes: they can disagree with humans, prefer whichever answer they see first, and reward longer answers even when the extra length adds nothing. If your judge is biased, every eval built on it is quietly wrong.

`judge-auditor` measures those failure modes on a labeled dataset and produces a report with confidence intervals.

## What it measures

| Audit | Question it answers | How |
|---|---|---|
| **Human agreement** | Does the judge pick the same winner as a human? | Accuracy and Cohen's kappa against human labels, with 95% bootstrap CIs |
| **Position bias** | Does the verdict change if we swap answer order? | Every pair is judged twice (A-first and B-first). Reports consistency and first-position win rate |
| **Verbosity bias** | Can filler make a wrong answer look better? | Pads the human-judged *worse* answer with content-free text and measures how much its score rises |

## Quick start

```bash
git clone https://github.com/b-mourya/judge-auditor.git
cd judge-auditor
pip install -e ".[dev]"

# Free: a simulated judge with deliberately planted biases
judge-audit --judge simulated

# Real judges (needs an API key)
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=...
judge-audit --judge anthropic --model claude-haiku-5-5 --out reports/haiku

pip install -e ".[openai]"
export OPENAI_API_KEY=...
judge-audit --judge openai --model <model-name> --out reports/openai
```

Each run writes `report.md` (human-readable) and `results.json` (every verdict, for further analysis). Real-judge calls are cached in `.cache/`, so re-running costs nothing.

An example report from the simulated judge is in [`examples/simulated/report.md`](examples/simulated/report.md).

## Dataset format

`data/sample_pairs.jsonl`, one comparison per line:

```json
{"id": "git-pull", "question": "What does git pull do?", "answer_a": "...", "answer_b": "...", "human": "A"}
```

`human` is `"A"`, `"B"` or `"tie"`. The 24 seed pairs cover programming, software testing, statistics and general knowledge. Several deliberately pair a short correct answer with a long, confident wrong one, which is exactly the situation where verbosity bias does damage.

## Why a simulated judge?

`SimulatedJudge` lets you dial in noise, position bias and length bias. The test suite uses it to prove each audit **detects the bias it targets and stays quiet when the bias is absent**. In other words, the auditor is itself tested.

## Project layout

```
src/judge_auditor/
  data.py      load and validate labeled pairs
  judges.py    judge prompt, verdict parsing, Anthropic/OpenAI/simulated judges, caching
  audits.py    agreement, position-bias and verbosity-bias audits
  stats.py     Cohen's kappa and bootstrap confidence intervals (stdlib only)
  report.py    Markdown report
  cli.py       `judge-audit` command
tests/         unit tests, including "does each audit catch its bias?" tests
```

## Roadmap

- [ ] Label 100+ pairs myself and report inter-annotator agreement
- [ ] Self-preference bias: does a judge favor answers written by its own model family?
- [ ] Mitigations: show how much swapping-and-averaging or a rubric prompt reduces each bias
- [ ] Compare several judge models side by side in one report
- [ ] Charts of results

## License

MIT
