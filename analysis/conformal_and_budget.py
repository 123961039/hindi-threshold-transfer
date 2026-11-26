"""Finite-sample / conformal source-max calibration and budget scaling analysis.

Drop this file into `analysis/conformal_and_budget.py` inside the
`hindi-threshold-transfer` repository and run from the repository root:

    python analysis/conformal_and_budget.py

Outputs:
    analysis/conformal_comparison.csv  (n=100, alpha=0.05 rule comparison)
    analysis/budget_scaling.csv        (n in {30, 60, 90, 100, 150, 210, 300})
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "scores.npz"
OUT_DIR = Path(__file__).resolve().parent

ARMS = ("setting_A_high2low", "setting_Bsm_leaveout_hi")
SOURCES = ("original", "bbc", "bhaav")
SEEDS = (42, 123, 2024)
N_DRAWS = 500
ALPHA = 0.05
RNG_SEED = 2095


def load_scores(path: Path) -> dict[tuple[str, str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Run from the repo root.")
    raw = np.load(path, allow_pickle=False)
    expected = {"original": 300, "bbc": 300, "bhaav": 206, "ai": 300}
    scores: dict[tuple[str, str], np.ndarray] = {}
    for arm in ARMS:
        for src, count in expected.items():
            arr = raw[f"{arm}__{src}"]
            assert arr.shape == (len(SEEDS), count), (arm, src, arr.shape)
            assert np.all((arr >= 0.0) & (arr <= 1.0)), (arm, src)
            scores[(arm, src)] = arr
    return scores


def order_stat_index_conformal(m: int, alpha: float) -> int:
    """0-based index for split-conformal quantile: min(m, ceil((m+1)*(1-alpha))) - 1."""
    k_1based = min(m, math.ceil((m + 1) * (1.0 - alpha)))
    return max(0, k_1based - 1)


def order_stat_index_ucb(m: int, alpha: float, delta: float) -> tuple[int, float]:
    """Smallest 0-based order-statistic index k such that P(FPR > alpha) <= delta.

    Under exchangeability, true FPR of S_(k_1based) follows Beta(m - k_1based + 1, k_1based),
    so P(FPR > alpha) = sum_{j=0}^{m - k_1based} C(m, j) alpha^j (1 - alpha)^(m - j).
    If even the maximum (k_1based = m) exceeds delta, returns m - 1 and its exact risk.
    """
    cum = 0.0
    best_idx = m - 1
    best_risk = (1.0 - alpha) ** m
    for r in range(m):  # r = m - k_1based, from 0 (max) upwards
        prob = math.comb(m, r) * (alpha**r) * ((1.0 - alpha) ** (m - r))
        cum += prob
        if cum <= delta:
            best_idx = (m - r) - 1
            best_risk = cum
        else:
            break
    return best_idx, best_risk


def split_budget(total: int, k: int = 3) -> tuple[int, ...]:
    return tuple(total // k + (1 if i < total % k else 0) for i in range(k))


def compute_threshold(
    cal_by_src: dict[str, np.ndarray], rule: str, alpha: float
) -> np.ndarray:
    """Return threshold array of shape (n_seeds,) for a given rule."""
    if rule == "pooled_emp":
        pooled = np.concatenate([cal_by_src[s] for s in SOURCES], axis=1)
        return np.quantile(pooled, 1.0 - alpha, axis=1, method="higher")

    per_src_thresholds = []
    for s in SOURCES:
        arr = cal_by_src[s]  # shape (n_seeds, m)
        m = arr.shape[1]
        if rule == "source_max_emp":
            t = np.quantile(arr, 1.0 - alpha, axis=1, method="higher")
        else:
            sorted_arr = np.sort(arr, axis=1)
            if rule == "source_max_conformal":
                idx = order_stat_index_conformal(m, alpha)
            elif rule == "source_max_bonferroni":
                idx = order_stat_index_conformal(m, alpha / len(SOURCES))
            elif rule == "source_max_ucb_d10":
                idx, _ = order_stat_index_ucb(m, alpha, delta=0.10)
            elif rule == "source_max_ucb_d05":
                idx, _ = order_stat_index_ucb(m, alpha, delta=0.05)
            elif rule == "source_max_ucb_bonf_d10":
                idx, _ = order_stat_index_ucb(m, alpha, delta=0.10 / len(SOURCES))
            else:
                raise ValueError(f"Unknown rule: {rule}")
            t = sorted_arr[:, idx]
        per_src_thresholds.append(t)
    return np.max(np.stack(per_src_thresholds, axis=0), axis=0)


def evaluate_rules(
    scores: dict[tuple[str, str], np.ndarray],
    budgets: list[int],
    rules: list[str],
    alpha: float = ALPHA,
) -> list[dict[str, object]]:
    """Evaluate rules across budgets on identical held-out test splits (200/200/106)."""
    rng = np.random.default_rng(RNG_SEED)
    lengths = {s: scores[(ARMS[0], s)].shape[1] for s in SOURCES}
    # Shared 500 permutations matching calibrate.py / joint_targets.py (first 100 = cal pool, rest = test)
    perms = [{s: rng.permutation(lengths[s]) for s in SOURCES} for _ in range(N_DRAWS)]

    rows: list[dict[str, object]] = []
    for budget in budgets:
        alloc = dict(zip(SOURCES, split_budget(budget, len(SOURCES))))
        if max(alloc.values()) > 100:
            raise ValueError(f"Per-source allocation {alloc} exceeds shared 100-doc cal pool.")

        for arm in ARMS:
            ai_scores = scores[(arm, "ai")]
            for rule in rules:
                fprs = {s: np.empty((N_DRAWS, len(SEEDS))) for s in SOURCES}
                recalls = np.empty((N_DRAWS, len(SEEDS)))

                for d, p in enumerate(perms):
                    cal_by_src = {
                        s: scores[(arm, s)][:, p[s][: alloc[s]]] for s in SOURCES
                    }
                    tau = compute_threshold(cal_by_src, rule, alpha)  # (n_seeds,)
                    for s in SOURCES:
                        test_s = scores[(arm, s)][:, p[s][100:]]
                        fprs[s][d] = (test_s > tau[:, None]).mean(axis=1)
                    recalls[d] = (ai_scores > tau[:, None]).mean(axis=1)

                stacked = np.stack([fprs[s] for s in SOURCES], axis=-1)  # (500, 3, 3)
                worst = stacked.max(axis=-1)  # (500, 3)
                joint_pass = worst <= alpha
                seed_passes = joint_pass.sum(axis=0)

                rows.append(
                    {
                        "budget": budget,
                        "per_source": "/".join(str(alloc[s]) for s in SOURCES),
                        "model": arm,
                        "rule": rule,
                        "orig_fpr_pct": round(100.0 * float(fprs["original"].mean()), 2),
                        "bbc_fpr_pct": round(100.0 * float(fprs["bbc"].mean()), 2),
                        "bhaav_fpr_pct": round(100.0 * float(fprs["bhaav"].mean()), 2),
                        "max_corpus_mean_fpr_pct": round(
                            100.0 * max(float(fprs[s].mean()) for s in SOURCES), 2
                        ),
                        "ai_recall_pct": round(100.0 * float(recalls.mean()), 2),
                        "joint_passes": int(joint_pass.sum()),
                        "seed_draws": N_DRAWS * len(SEEDS),
                        "joint_rate_pct": round(100.0 * float(joint_pass.mean()), 1),
                        "worst_fpr_p95_pct": round(
                            100.0 * float(np.quantile(worst, 0.95)), 1
                        ),
                        "seed_passes": "/".join(str(int(x)) for x in seed_passes),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    scores = load_scores(DATA_PATH)

    # 1) Rule comparison at n=100 (same 34/33/33 split as Tables 3 & 4)
    rules_n100 = [
        "pooled_emp",
        "source_max_emp",
        "source_max_conformal",
        "source_max_bonferroni",
        "source_max_ucb_d10",
        "source_max_ucb_d05",
    ]
    rows_n100 = evaluate_rules(scores, budgets=[100], rules=rules_n100)
    out1 = OUT_DIR / "conformal_comparison.csv"
    write_csv(out1, rows_n100)

    print("=== Part A: Rule Comparison at n=100 (alpha=5%) ===")
    for r in rows_n100:
        print(
            f"{r['model']:24s} | {r['rule']:22s} | "
            f"Orig={r['orig_fpr_pct']:5.2f}% BBC={r['bbc_fpr_pct']:4.2f}% "
            f"BHAAV={r['bhaav_fpr_pct']:5.2f}% | Rec={r['ai_recall_pct']:5.2f}% | "
            f"Joint={r['joint_passes']:4d}/{r['seed_draws']} ({r['joint_rate_pct']:5.1f}%) | "
            f"MaxP95={r['worst_fpr_p95_pct']:4.1f}%"
        )

    # 2) Budget scaling curve on identical held-out test splits (200/200/106)
    budgets = [30, 60, 90, 100, 150, 210, 300]
    rules_budget = [
        "pooled_emp",
        "source_max_emp",
        "source_max_conformal",
        "source_max_bonferroni",
        "source_max_ucb_d10",
    ]
    rows_budget = evaluate_rules(scores, budgets=budgets, rules=rules_budget)
    out2 = OUT_DIR / "budget_scaling.csv"
    write_csv(out2, rows_budget)

    print("\n=== Part B: Budget Scaling (Bsm, identical 200/200/106 test splits) ===")
    for r in rows_budget:
        if r["model"] == "setting_Bsm_leaveout_hi":
            print(
                f"n={r['budget']:3d} ({r['per_source']:>10s}) | {r['rule']:22s} | "
                f"MaxMean={r['max_corpus_mean_fpr_pct']:5.2f}% | Rec={r['ai_recall_pct']:5.2f}% | "
                f"Joint={r['joint_rate_pct']:5.1f}% | MaxP95={r['worst_fpr_p95_pct']:4.1f}%"
            )

    print(f"\nSaved: {out1} and {out2}")


if __name__ == "__main__":
    main()