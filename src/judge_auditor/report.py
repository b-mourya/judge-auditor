"""Turn audit results into a readable Markdown report."""
from __future__ import annotations


def _ci(ci) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "n/a"


def _kappa_band(k: float) -> str:
    if k < 0.2:
        return "slight"
    if k < 0.4:
        return "fair"
    if k < 0.6:
        return "moderate"
    if k < 0.8:
        return "substantial"
    return "almost perfect"


def render(judge_name: str, dataset: str, agreement: dict, position: dict, verbosity: dict) -> str:
    flags = []
    if agreement["kappa"] < 0.6:
        flags.append(f"Human agreement is only {_kappa_band(agreement['kappa'])} (kappa {agreement['kappa']:.2f}).")
    if position["consistency"] < 0.9:
        flags.append(f"Verdict changes in {1 - position['consistency']:.0%} of pairs when answer order is swapped.")
    vd, vci = verbosity.get("mean_delta"), verbosity.get("mean_delta_ci")
    if vd is not None and vci[0] > 0:
        flags.append(f"Padding the worse answer with filler raises its score by {vd:+.2f} (CI excludes 0).")

    fpw = position["first_position_win_rate"]
    lines = [
        "# LLM-as-Judge Audit Report",
        "",
        f"**Judge:** `{judge_name}`  ",
        f"**Dataset:** `{dataset}` ({agreement['n']} labeled pairs)",
        "",
        "## Summary",
        "",
        *([f"- ⚠️ {f}" for f in flags] or ["- ✅ No major issues detected on this dataset."]),
        "",
        "## 1. Agreement with human labels",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
        f"| Accuracy | {agreement['accuracy']:.2f} | {_ci(agreement['accuracy_ci'])} |",
        f"| Cohen's kappa | {agreement['kappa']:.2f} ({_kappa_band(agreement['kappa'])}) | {_ci(agreement['kappa_ci'])} |",
        "",
        "Kappa corrects for agreement that would happen by chance. 0 = chance level, 1 = perfect.",
        "",
    ]
    if agreement["disagreements"]:
        lines += ["Disagreements: " + ", ".join(
            f"`{d['id']}` (human {d['human']}, judge {d['judge']})" for d in agreement["disagreements"]), ""]

    lines += [
        "## 2. Position bias",
        "",
        "Each pair is judged twice, once with A shown first and once with B shown first.",
        "",
        "| Metric | Value | Ideal |",
        "|---|---|---|",
        f"| Consistency after swap | {position['consistency']:.2f} {_ci(position['consistency_ci'])} | 1.00 |",
        f"| First-position win rate | {fpw:.2f} | 0.50 |" if fpw is not None else "| First-position win rate | n/a | 0.50 |",
        "",
    ]
    if position["inconsistent"]:
        lines += ["Inconsistent pairs: " + ", ".join(
            f"`{d['id']}` (A-first: {d['a_first']}, B-first: {d['b_first']})" for d in position["inconsistent"]), ""]

    lines += [
        "## 3. Verbosity bias",
        "",
        "The answer humans judged *worse* is padded with filler that adds no information. "
        "Loser score = 1 if the judge picks it, 0.5 for a tie, 0 otherwise (averaged over both orders).",
        "",
    ]
    if vd is None:
        lines += ["No decisive human labels, so this audit was skipped.", ""]
    else:
        lines += [
            "| Metric | Value |",
            "|---|---|",
            f"| Loser score before padding | {verbosity['loser_score_before']:.2f} |",
            f"| Loser score after padding | {verbosity['loser_score_after']:.2f} |",
            f"| Change | {vd:+.2f} {_ci(vci)} |",
            "",
        ]
        if verbosity["flipped"]:
            lines += ["Pairs where padding rescued a clearly losing answer: "
                      + ", ".join(f"`{i}`" for i in verbosity["flipped"]), ""]
    return "\n".join(lines)
