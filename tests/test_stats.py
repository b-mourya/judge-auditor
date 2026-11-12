from judge_auditor.stats import bootstrap_ci, cohens_kappa, mean


def test_kappa_perfect_agreement():
    assert cohens_kappa(["A", "B", "tie"], ["A", "B", "tie"]) == 1.0


def test_kappa_known_value():
    # Two raters, 50 items. Classic textbook example: observed 0.7, expected 0.5 -> kappa 0.4
    a = ["A"] * 25 + ["B"] * 25
    b = ["A"] * 20 + ["B"] * 5 + ["A"] * 10 + ["B"] * 15
    assert abs(cohens_kappa(a, b) - 0.4) < 1e-9


def test_kappa_single_label_edge_case():
    assert cohens_kappa(["A", "A"], ["A", "A"]) == 1.0


def test_bootstrap_ci_contains_mean():
    xs = [0.0, 1.0] * 50
    lo, hi = bootstrap_ci(xs, mean, iters=500)
    assert lo <= 0.5 <= hi
