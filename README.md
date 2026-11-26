# Hindi Threshold Transfer

Reproduction artifact for the Hindi cross-corpus AI-text detection audit. The three scripts reproduce the saved-score analysis without GPUs, model downloads, or original documents. The bundled 64-bit score arrays retain alignment and within-source ordering across checkpoints. Each array contains only model scores; source document IDs and URLs are absent.

## Reproduction

Use Python 3.11+ and NumPy. From the repository root:

```bash
python -m pip install -r requirements.txt
python analysis/calibrate.py
python analysis/extra_analysis.py
python analysis/joint_targets.py
```

| Output | Coverage |
| --- | --- |
| `analysis/calibration_sensitivity.csv` | Calibration allocations, 50/100 budgets, held-out FPR, per-seed means, partition percentiles and original-benchmark recall (Tables 1, 2, 5, 6). |
| `analysis/extra_calibration.csv` | Pooled versus source-max cutoffs at 1%, 2%, 5%, and 10% targets (Table 3). |
| `analysis/joint_targets.csv` | Three-corpus simultaneous 5% attainment and per-pair worst-source FPR (Table 4). |

Numerical checkpoints: BBC-only calibration on Bsm yields 4.92% BBC FPR, 39.95% BHAAV FPR and 46.21% original-corpus FPR at budget 100. The pooled rule passes 258 of 1,500 checkpoint–split pairs. The source-max rule passes 963 of 1,500.

## Score layout and scope

`data/scores.npz` keys use `training_configuration__evaluation_source`. Rows are seeds 42, 123 and 2024. Within each source the columns align across training configurations and seeds. Human source sizes are original CC100-derived benchmark (300), BBC Hindi (300), and BHAAV (206). The `ai` source contains 300 generated original-benchmark examples. All recall results refer to that benchmark. The artifact starts from frozen predictions. Texts, weights, training logs and audit responses are absent.

The protocol uses 500 paired splits and a strict comparison against the empirical quantile (NumPy `higher`).
