import csv
from pathlib import Path
import numpy as np

sources = ('original', 'bbc', 'bhaav')
arms = ('setting_A_high2low', 'setting_Bsm_leaveout_hi')
seeds = (42, 123, 2024)
root = Path(__file__).resolve().parents[1]
with np.load(root / 'data/scores.npz', allow_pickle=False) as dataset:
    expected = {f"{arm}__{source}" for arm in arms for source in (*sources, 'ai')}
    if set(dataset.files) != expected:
        raise ValueError("Score archive columns differ from the evaluation protocol")
    scores = {
        arm: {source: dataset[f"{arm}__{source}"] for source in (*sources, 'ai')}
        for arm in arms
    }
for matrix in scores.values():
    for source, values in matrix.items():
        count = 206 if source == 'bhaav' else 300
        if values.shape != (len(seeds), count) or not np.all((0 <= values) & (values <= 1)):
            raise ValueError(f"Invalid score array for {source}")

def evaluate(matrix, budget):
    rng = np.random.default_rng(2095)
    allocation = {key: tuple(budget if key == x else 0 for x in sources) for key in sources}
    allocation['mix_all'] = tuple(budget // 3 + (i < budget % 3) for i in range(3))
    for target in sources:
        allocation[f'without_{target}'] = tuple(0 if x == target else budget // 2 for x in sources)
    results = {key: [] for key in allocation}
    for _ in range(500):
        perm = [rng.permutation(matrix[x].shape[1]) for x in sources]
        test = [matrix[x][:, p[budget:]] for x, p in zip(sources, perm)]
        for policy, counts in allocation.items():
            calibration = np.concatenate(
                [matrix[x][:, p[:n]] for x, p, n in zip(sources, perm, counts)], axis=1)
            threshold = np.quantile(calibration, .95, axis=1, method='higher')[:, None]
            results[policy].append(np.column_stack([(v > threshold).mean(axis=1)
                                                     for v in (*test, matrix['ai'])]))
    return results

def main():
    output = []
    for arm, matrix in scores.items():
        for budget in (50, 100):
            for policy, draws in evaluate(matrix, budget).items():
                values = np.array(draws) * 100
                for i, metric in enumerate((*sources, 'ai')):
                    by_draw = values[:, :, i].mean(axis=1)
                    lo, hi = np.quantile(by_draw, (.025, .975))
                    output.append([arm, budget, policy, metric, by_draw.mean(), lo, hi,
                                   *values[:, :, i].mean(axis=0)])
    with Path(__file__).with_name('calibration_sensitivity.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm','budget','policy','evaluation_source','mean_pct',
                    'draw_p025_pct','draw_p975_pct','seed42_pct','seed123_pct','seed2024_pct'])
        w.writerows(output)


if __name__ == '__main__':
    main()
