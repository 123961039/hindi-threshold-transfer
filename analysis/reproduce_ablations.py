"""Pure-CPU reproduction of length/punctuation ablations and zero-shot Binoculars transfer.

Reads:
    data/ablation_scores.npz
    data/binoculars_neg_scores.npz

Outputs:
    analysis/ablation_summary.csv
    analysis/binoculars_transfer_summary.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent

ARMS = ("setting_A_high2low", "setting_Bsm_leaveout_hi")
VARIANTS = ("raw", "len50", "punct_danda", "len50_punct_danda")
SOURCES = ("original", "bbc", "bhaav")
RNG_SEED = 2095
N_DRAWS = 500
BUDGET = 100
ALPHA = 0.05


def evaluate_transfer(score_map: dict[str, np.ndarray]) -> list[dict[str, object]]:
    n_seeds = score_map["original"].shape[0]
    rng = np.random.default_rng(RNG_SEED)
    perms = [
        {s: rng.permutation(score_map[s].shape[1]) for s in SOURCES}
        for _ in range(N_DRAWS)
    ]

    alloc_mix = tuple(BUDGET // 3 + (1 if i < BUDGET % 3 else 0) for i in range(3))
    alloc_half = (BUDGET // 2, BUDGET - BUDGET // 2)
    ai_scores = score_map["ai"]

    policies = [
        "bbc_only",
        "orig_only",
        "bhaav_only",
        "omit_original",
        "omit_bhaav",
        "all_three_pooled",
        "all_three_source_max",
    ]

    results: list[dict[str, object]] = []
    if float(np.nanmin(score_map["original"])) >= 0.0 and float(np.nanmax(score_map["original"])) <= 1.0:
        results.append(
            {
                "policy": "default_0.5",
                "orig_fpr_pct": round(100.0 * float((score_map["original"] >= 0.5).mean()), 2),
                "bbc_fpr_pct": round(100.0 * float((score_map["bbc"] >= 0.5).mean()), 2),
                "bhaav_fpr_pct": round(100.0 * float((score_map["bhaav"] >= 0.5).mean()), 2),
                "ai_recall_pct": round(100.0 * float((ai_scores >= 0.5).mean()), 2),
                "joint_rate_pct": "-",
                "worst_p95_pct": "-",
            }
        )

    for pol in policies:
        fprs = {s: np.empty((N_DRAWS, n_seeds)) for s in SOURCES}
        recalls = np.empty((N_DRAWS, n_seeds))

        for d, p in enumerate(perms):
            if pol == "bbc_only":
                tau = np.quantile(score_map["bbc"][:, p["bbc"][:BUDGET]], 1.0 - ALPHA, axis=1, method="higher")
            elif pol == "orig_only":
                tau = np.quantile(score_map["original"][:, p["original"][:BUDGET]], 1.0 - ALPHA, axis=1, method="higher")
            elif pol == "bhaav_only":
                tau = np.quantile(score_map["bhaav"][:, p["bhaav"][:BUDGET]], 1.0 - ALPHA, axis=1, method="higher")
            elif pol == "omit_original":
                cal = np.concatenate(
                    [score_map["bbc"][:, p["bbc"][: alloc_half[0]]], score_map["bhaav"][:, p["bhaav"][: alloc_half[1]]]],
                    axis=1,
                )
                tau = np.quantile(cal, 1.0 - ALPHA, axis=1, method="higher")
            elif pol == "omit_bhaav":
                cal = np.concatenate(
                    [score_map["original"][:, p["original"][: alloc_half[0]]], score_map["bbc"][:, p["bbc"][: alloc_half[1]]]],
                    axis=1,
                )
                tau = np.quantile(cal, 1.0 - ALPHA, axis=1, method="higher")
            elif pol == "all_three_pooled":
                cal = np.concatenate([score_map[s][:, p[s][: alloc_mix[i]]] for i, s in enumerate(SOURCES)], axis=1)
                tau = np.quantile(cal, 1.0 - ALPHA, axis=1, method="higher")
            elif pol == "all_three_source_max":
                taus = [
                    np.quantile(score_map[s][:, p[s][: alloc_mix[i]]], 1.0 - ALPHA, axis=1, method="higher")
                    for i, s in enumerate(SOURCES)
                ]
                tau = np.max(np.stack(taus, axis=0), axis=0)
            else:
                raise ValueError(pol)

            for s in SOURCES:
                test_s = score_map[s][:, p[s][BUDGET:]]
                fprs[s][d] = (test_s > tau[:, None]).mean(axis=1)
            recalls[d] = (ai_scores > tau[:, None]).mean(axis=1)

        worst = np.stack([fprs[s] for s in SOURCES], axis=-1).max(axis=-1)
        results.append(
            {
                "policy": pol,
                "orig_fpr_pct": round(100.0 * float(fprs["original"].mean()), 2),
                "bbc_fpr_pct": round(100.0 * float(fprs["bbc"].mean()), 2),
                "bhaav_fpr_pct": round(100.0 * float(fprs["bhaav"].mean()), 2),
                "ai_recall_pct": round(100.0 * float(recalls.mean()), 2),
                "joint_rate_pct": round(100.0 * float((worst <= ALPHA).mean()), 1),
                "worst_p95_pct": round(100.0 * float(np.quantile(worst, 0.95)), 1),
            }
        )
    return results


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ablation_path = DATA_DIR / "ablation_scores.npz"
    if ablation_path.exists():
        raw = np.load(ablation_path, allow_pickle=False)
        rows = []
        print("=== Length & Punctuation Ablation (Bsm, n=100, alpha=5%) ===")
        for arm in ARMS:
            for var in VARIANTS:
                smap = {src: raw[f"{arm}__{var}__{src}"] for src in ("original", "bbc", "bhaav", "ai")}
                for r in evaluate_transfer(smap):
                    row = {"model": arm, "variant": var, **r}
                    rows.append(row)
                    if arm == "setting_Bsm_leaveout_hi" and r["policy"] in (
                        "bbc_only",
                        "all_three_pooled",
                        "all_three_source_max",
                    ):
                        print(
                            f"Bsm | {var:18s} | {r['policy']:20s} | "
                            f"Orig={r['orig_fpr_pct']:5.2f}% BBC={r['bbc_fpr_pct']:5.2f}% "
                            f"BHAAV={r['bhaav_fpr_pct']:5.2f}% | Rec={r['ai_recall_pct']:5.2f}% | "
                            f"Joint={str(r['joint_rate_pct']):>5s}%"
                        )
        out_ab = OUT_DIR / "ablation_summary.csv"
        write_csv(out_ab, rows)
        print(f"Saved: {out_ab}")
    else:
        print(f"[skip] {ablation_path} not found.")

    bino_path = DATA_DIR / "binoculars_neg_scores.npz"
    if bino_path.exists():
        raw_b = np.load(bino_path, allow_pickle=False)
        smap_b = {src: raw_b[src] for src in ("original", "bbc", "bhaav", "ai")}
        rows_b = evaluate_transfer(smap_b)
        print("\n=== Zero-Shot Binoculars Cross-Corpus Transfer (n=100, alpha=5%) ===")
        for r in rows_b:
            print(
                f"Binoculars | {r['policy']:20s} | "
                f"Orig={r['orig_fpr_pct']:5.2f}% BBC={r['bbc_fpr_pct']:5.2f}% "
                f"BHAAV={r['bhaav_fpr_pct']:5.2f}% | Rec={r['ai_recall_pct']:5.2f}% | "
                f"Joint={str(r['joint_rate_pct']):>5s}% | MaxP95={str(r['worst_p95_pct']):>5s}%"
            )
        out_b = OUT_DIR / "binoculars_transfer_summary.csv"
        write_csv(out_b, rows_b)
        print(f"Saved: {out_b}")
    else:
        print(f"[skip] {bino_path} not found.")


if __name__ == "__main__":
    main()
