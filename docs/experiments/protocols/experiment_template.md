# Experiment: <ID — short title>

[Research notebook](../README.md)

Copy this file for a new run. Fill in the plan first; use `not measured` for missing results. Keep failed and interrupted runs.

## Question

- **Hypothesis:** <what might explain the result?>
- **Prediction:** <metric, group, and expected direction>
- **What would disprove it:** <specific result>
- **Motivation:** <link to earlier evidence>
- **Status and dates:** <planned / running / complete / interrupted; UTC>

## Setup

| Field | Value |
|---|---|
| Dataset and hashes | <source, preprocessing, files> |
| Split and targets | <how train/validation/test are defined> |
| Users | <count, full or sampled, sample seed> |
| Catalog | <warm or full, size, ID mapping> |
| Popularity groups | <training-count thresholds> |
| Filtering | <seen items, repeated targets, PAD> |
| Retrieval | <beam or exact, width, scoring/normalization> |
| Checkpoints and SIDs | <files, hashes, epochs, selection metric> |
| Training | <loss, hyperparameters, seed, epochs> |
| Evaluation | <metrics and K values, target groups> |

## Comparison

**Baseline:** <the unchanged reference condition>

**Treatment:** <one change from the baseline>

**Other differences or confounders:** <any unavoidable changes>

**Run commands:**

```bash
# Paste the exact commands here.
```

**Output files:** <checkpoint, history, metrics, report>

## Results

| Group | n | Baseline R@10 | Treatment R@10 | Baseline R@50 | Treatment R@50 |
|---|---:|---:|---:|---:|---:|
| All | | | | | |
| Head | | | | | |
| Mid | | | | | |
| Tail | | | | | |
| Unseen | | | | | |

Include NDCG, prefix survival, runtime, or other metrics only when relevant. Report counts, epochs, each seed, and timing conditions if measured. Do not treat a smoke run as a full result.

## Conclusion

- **Observed:** <what the saved outputs show>
- **Possible explanation:** <and other plausible causes>
- **Limitations:** <single seed, subsets, missing metadata, reused test data>
- **Decision:** <keep / revert / repeat / next experiment>
- **Evidence:** <artifact paths, hashes, notebook links>
