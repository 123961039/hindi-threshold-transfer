# Hindi Threshold Transfer

Reproduction artifact for the Hindi cross-corpus AI-text detection audit. All scripts reproduce the saved-score analyses on CPU using NumPy without GPUs, model downloads, or copyrighted source texts. Each score array contains only model outputs aligned within each corpus across checkpoints; raw documents are omitted.

## Reproduction

Use Python 3.11+ and NumPy. From the repository root:

```bash
python -m pip install -r requirements.txt
python analysis/calibrate.py
python analysis/extra_analysis.py
python analysis/joint_targets.py
python analysis/conformal_and_budget.py
python analysis/reproduce_ablations.py
```

| Output | Coverage |
| --- | --- |
| `analysis/calibration_sensitivity.csv` | Calibration allocations, 50/100 budgets, held-out FPR, per-seed means, partition percentiles, and original-benchmark recall (Tables 1, 2, 5, 6). |
| `analysis/extra_calibration.csv` | Pooled versus source-max cutoffs at 1%, 2%, 5%, and 10% targets (Table 3). |
| `analysis/joint_targets.csv` | Three-corpus simultaneous 5% attainment and per-pair worst-source FPR (Table 4). |
| `analysis/conformal_comparison.csv` | Finite-sample conformal, Bonferroni, and Beta/Clopper–Pearson UCB source-max comparison at `n = 100` (`alpha = 5%`). |
| `analysis/budget_scaling.csv` | Budget scaling curve (`n` from 30 to 300) on identical `200 / 200 / 106` held-out test partitions. |
| `analysis/ablation_summary.csv` | Length-matched (50 words) and Devanagari danda punctuation-normalized ablations across all six frozen XLM-R-large checkpoints. |
| `analysis/binoculars_transfer_summary.csv` | Zero-shot Binoculars (`Qwen2.5-0.5B` / `Qwen2.5-0.5B-Instruct`) cross-corpus threshold transfer across Original, BBC, and BHAAV. |

### Numerical checkpoints

- **Main transfer (`Bsm`, `n = 100`, 5% target)**: BBC-only calibration yields `4.92%` BBC FPR, `39.95%` BHAAV FPR, and `46.21%` Original FPR.
- **Joint attainment (`Bsm`, `n = 100`, 1,500 checkpoint–split pairs)**: `pooled_emp` passes `258 / 1,500` (`17.2%`), `source_max_emp` passes `963 / 1,500` (`64.2%`), and `source_max_conformal` passes `1,406 / 1,500` (`93.7%`, `MaxP95 = 5.5%`, recall `95.23%`).
- **Budget scaling (`Bsm`, `n = 300`, 100 per source)**: `pooled_emp` drops to `5.3%` joint attainment, whereas `source_max_ucb_d10` reaches `97.7%` (`MaxP95 = 4.7%`, recall `97.51%`).
- **Length + punctuation ablation (`Bsm`, `len50_punct_danda`, BBC-only)**: Orig `19.31%`, BBC `5.02%`, BHAAV `17.32%`, Joint `0.0%`; `source_max` lowers all three below 5% (`3.42% / 0.23% / 3.94%`, Joint `65.1%`).
- **Zero-shot Binoculars (`n = 100`, 5% target)**: BBC-only yields `4.90%` BBC, `17.46%` Orig, `18.18%` BHAAV (`Joint = 0.0%`); `source_max` yields `4.77%` Orig, `0.26%` BBC, `2.70%` BHAAV (`Joint = 62.4%`).

## Data layout and scope

- `data/scores.npz`: keys `training_configuration__evaluation_source` for `setting_A_high2low` and `setting_Bsm_leaveout_hi` across seeds `42, 123, 2024` and corpora `original` (300), `bbc` (300), `bhaav` (206), and `ai` (300).
- `data/ablation_scores.npz`: keys `training_configuration__variant__evaluation_source` for `raw`, `len50`, `punct_danda`, and `len50_punct_danda`.
- `data/binoculars_neg_scores.npz`: negated Binoculars scores (`original`, `bbc`, `bhaav`, `ai`; higher = more AI-like).
- `metadata/` (if present): frozen BHAAV boundary-audit table (`bhaav_boundaries.csv`) and SHA-256 input hashes (`bhaav_inputs.sha256`).

All protocols use 500 paired splits (`np.random.default_rng(2095)`) and strict `s > tau` flagging against the empirical `higher` quantile unless noted otherwise.
