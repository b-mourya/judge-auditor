# LLM-as-Judge Audit Report

**Judge:** `simulated(noise=0.15, position_bias=0.25, length_bias=0.6)`  
**Dataset:** `data/sample_pairs.jsonl` (24 labeled pairs)

## Summary

- ⚠️ Human agreement is only moderate (kappa 0.51).
- ⚠️ Verdict changes in 50% of pairs when answer order is swapped.
- ⚠️ Padding the worse answer with filler raises its score by +0.29 (CI excludes 0).

## 1. Agreement with human labels

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 0.71 | [0.50, 0.88] |
| Cohen's kappa | 0.51 (moderate) | [0.27, 0.75] |

Kappa corrects for agreement that would happen by chance. 0 = chance level, 1 = perfect.

Disagreements: `py-len` (human B, judge tie), `australia` (human B, judge tie), `git-undo` (human B, judge tie), `json-quotes` (human B, judge tie), `ci` (human tie, judge B), `light` (human tie, judge A), `bit` (human B, judge tie)

## 2. Position bias

Each pair is judged twice, once with A shown first and once with B shown first.

| Metric | Value | Ideal |
|---|---|---|
| Consistency after swap | 0.50 [0.29, 0.67] | 1.00 |
| First-position win rate | 0.68 | 0.50 |

Inconsistent pairs: `http-404` (A-first: A, B-first: tie), `py-len` (A-first: tie, B-first: B), `binary-search` (A-first: A, B-first: tie), `australia` (A-first: tie, B-first: B), `git-undo` (A-first: tie, B-first: B), `where-having` (A-first: A, B-first: tie), `boiling` (A-first: tie, B-first: B), `json-quotes` (A-first: tie, B-first: B), `km-miles` (A-first: A, B-first: tie), `light` (A-first: A, B-first: B), `bit` (A-first: tie, B-first: B), `https-port` (A-first: A, B-first: tie)

## 3. Verbosity bias

The answer humans judged *worse* is padded with filler that adds no information. Loser score = 1 if the judge picks it, 0.5 for a tie, 0 otherwise (averaged over both orders).

| Metric | Value |
|---|---|
| Loser score before padding | 0.12 |
| Loser score after padding | 0.40 |
| Change | +0.29 [0.21, 0.36] |

Pairs where padding rescued a clearly losing answer: `bva`, `false-positive`, `rag`, `prime-91`, `regression`, `austen`, `mutable-default`, `git-pull`, `p-value`, `equivalence`, `overfitting`
