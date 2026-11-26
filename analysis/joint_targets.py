"""Report joint 5% FPR attainment across sources using saved predictions."""
import csv
from pathlib import Path

import numpy as np
from calibrate import scores, sources

output = []
for arm, matrix in scores.items():
    rng = np.random.default_rng(2095)
    policies = {
        'bbc': [], 'pooled': [], 'source_max': [], 'omit_bhaav': [],
    }
    for _ in range(500):
        perm = [rng.permutation(matrix[source].shape[1]) for source in sources]
        tests = [matrix[source][:, idx[100:]] for source, idx in zip(sources, perm)]
        mixed = [matrix[source][:, idx[:count]]
                 for source, idx, count in zip(sources, perm, (34, 33, 33))]
        def quantile(values):
            return np.quantile(values, 0.95, axis=1, method='higher')
        thresholds = (
            quantile(matrix['bbc'][:, perm[1][:100]]),
            quantile(np.hstack(mixed)),
            np.max([quantile(values) for values in mixed], axis=0),
            quantile(np.hstack([matrix[source][:, idx[:50]]
                                for source, idx in zip(sources[:2], perm[:2])])),
        )
        for values, threshold in zip(policies.values(), thresholds):
            values.append(np.stack([(test > threshold[:, None]).mean(axis=1)
                                    for test in tests], axis=1))
    for policy, values in policies.items():
        worst = np.asarray(values).max(axis=-1)
        passes = worst <= 0.05
        output.append([arm, policy, int(passes.sum()), passes.size,
                       100 * float(np.quantile(worst, 0.95)),
                       *map(int, passes.sum(axis=0))])
with Path(__file__).with_name('joint_targets.csv').open('w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['arm', 'policy', 'passes', 'seed_draws',
                     'worst_fpr_p95_pct', 'passes_seed42', 'passes_seed123',
                     'passes_seed2024'])
    writer.writerows(output)
