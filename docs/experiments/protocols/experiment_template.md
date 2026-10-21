# Experiment: <ID — short descriptive title>

[Notebook index](../README.md)

Copy this file for a future experiment. Replace placeholders before running;
leave unavailable results as `not measured`. Record unsuccessful and interrupted
runs. Distinguish a plan, an implemented method, and a completed measurement.

## Question and prediction

- **Hypothesis:** <specific mechanism or limitation being tested>
- **Prediction:** <expected direction on a named metric and population>
- **Evidence against it:** <what outcome would weaken the hypothesis>
- **Reason to test now:** <previous observation and link to its evidence>
- **Status / dates:** <proposed, running, complete, interrupted; UTC timestamps>

## Evaluation contract

| Field | Value |
|---|---|
| Dataset / preprocessing | <categories, filters, sample, snapshot hashes> |
| Split | <per-user or global temporal; exact target/history definition> |
| Users / targets | <full population or subset; size and selection seed> |
| Candidate catalog | <warm or full; catalog identity and size> |
| Target groups | <all, warm, head/mid/tail/unseen; exact definitions> |
| Seen-item policy | <before/after generation; repeated-target eligibility> |
| Retrieval | <exact ranking or beam; beam width, score normalization, priors> |
| Primary selection metric | <validation metric chosen before inspection> |
| Secondary metrics | <Recall/NDCG cutoffs, group metrics, throughput, memory> |
| Units | <fractions or percentages; seconds; GiB> |
| Final test policy | <fixed selection and when test will be evaluated> |

## Control and treatment

| Setting | Baseline | Treatment |
|---|---|---|
| Training examples / weighting | <...> | <...> |
| Model dimensions / heads / layers | <...> | <...> |
| Features / tokenizer / SID source | <...> | <...> |
| Objective / negatives / sampling | <...> | <...> |
| Optimizer / LR / batch / schedule | <...> | <...> |
| Budget / patience / seeds | <...> | <...> |
| Other settings | <unchanged settings listed explicitly> | <...> |

**Intended isolated change:** <one change, or explicitly call this a combined
experiment>. If coverage or batch size changes, compare optimizer steps,
examples processed, and wall time as well as epochs.

## Reproduction and provenance

- Code revision and uncommitted diff: <commit plus preserved patch>
- Exact commands and working directory: <commands actually used>
- Environment: <Python, PyTorch/CUDA, dependencies, CPU/GPU>
- Immutable input/teacher/tokenizer/checkpoint paths and hashes: <...>
- Run IDs, config and history files: <...>
- Raw evaluation artifacts and hashes: <...>
- Changes or interruptions during execution: <...>

Do not pair a checkpoint with a different mapping or tokenizer just because the
filenames look compatible. Save actual weights per run when comparing models;
timestamped metric histories do not preserve overwritten checkpoints.

## Results

| Split / population / metric | Baseline | Treatment | Absolute delta |
|---|---:|---:|---:|
| <primary validation result> | not measured | not measured | — |
| <secondary metric> | not measured | not measured | — |
| <head / mid / tail / unseen> | not measured | not measured | — |
| <test after selection> | not measured | not measured | — |
| <training examples/s / time / peak memory> | not measured | not measured | — |

Include target counts and checkpoint epochs. Label training versus validation
loss; record aggregation. For repeated seeds report individual results and a
summary, with uncertainty where available. For timing, state warm-up, measured
region, synchronization, validation overhead, and whether GPU work overlapped.

## Interpretation and decision

- **Measured:** <what the artifacts establish>
- **Recorded only:** <discussion observations lacking retained raw evidence>
- **Interpretation:** <mechanism consistent with the result>
- **Alternatives / confounders:** <other plausible explanations>
- **Conclusion:** <supported, weakened, mixed, or inconclusive for this test>
- **Decision:** <keep, revert, pause, repeat, or next isolated experiment>
- **Limitations:** <subset, single seed, missing provenance, reused test set>
- **Notebook updates:** <hypothesis index, result table, evidence manifest links>

An implementation that runs is not yet an efficacy result. A negative result on
one setting is not a universal rejection of the approach.
