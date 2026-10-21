# Baseline development: correctness, quality, and throughput

[Notebook index](../README.md) · [Later research](../research_log.md) ·
[Archived results](../archive.md) · [Machine-readable evidence](../../../artifacts/main/derived/baseline_evidence.json)

Reviewed **2026-09-19**. This chapter preserves the September 8 discussion and
reconciles it with the surviving histories and current implementation. It fills
in the baseline work preceding the SID experiments. No model was trained or
re-evaluated to write this chapter.

**Metric convention:** this page uses fractions: `0.03 = 3%`. The other notebook
results tables generally use percentages. Unless stated otherwise, baseline
Recall@20 uses warm candidates and warm targets. It is not interchangeable with
GenRec Recall@10 over every target and full-catalog beam search.

## Objective and status

The user's quality target was **Recall@20 ≥ 0.10**, starting from approximately
0.03 for the content Two-Tower. The completed baseline runs here do not reach it.
The largest saved validation selection score is 0.037466; its warm-target test
Recall@20 is 0.030925. These are different splits, not two estimates of one score.

| ID | Hypothesis / question | Evidence and current interpretation |
|---|---|---|
| B01 | Near-random SASRec recall reflects failed learning | Incorrect as the initial explanation: evaluation produced NaNs; a same-checkpoint diagnostic recovered much higher recall |
| B02 | Timestamp sorting preserves the leave-two-out split | False for ties: preserve the original interaction `position` |
| B03 | Greater width, more heads, or greater depth improves quality | Mixed results; width is not monotonically beneficial, and the 256-d Two-Tower run is incomplete |
| B04 | All training prefixes improve coverage and quality | Coverage increase implemented and counted; quality comparison also changed batch size and epoch budget |
| B05 | More negatives / a different ranking objective improves retrieval | Both current baselines use 64 negatives; no isolated 64/256/1,024-negative sweep recovered |
| B06 | A learned item residual can complement content features | Proposed for Two-Tower; no attributable baseline experiment recovered |
| B07 | More training users for the retained catalog improve sparse items | Proposed; no controlled data-expansion experiment recovered |
| B08 | CPU preparation and synchronization cause GPU idle time | Loader benchmark supports preparation overhead; user reported utilization increasing to 99%; no matched full training speedup measured |
| B09 | Durable histories make comparisons and plots possible | Implemented for both baselines; timestamped histories retain config, epochs, and final metrics |

Use **Measured** for retained artifacts or calculations; **Recorded** for output
preserved only in the discussion; **Implemented** for current code; **Proposed**
for untested ideas; and **Interpretation** for explanations. A suggested setting
is not a completed experiment, and an interrupted run is not a failed hypothesis.

## Dataset and assumptions

The pipeline loads only three category subsets of Amazon Reviews 2023. Selecting
a user does not fetch that user's events from other categories. A sequence can
mix the selected categories. Products may have overlapping metadata labels;
the inclusion rule is the source interaction subset.

| Source category | Retained interactions | Retained items |
|---|---:|---:|
| Electronics | 657,997 | 97,683 |
| Cell Phones and Accessories | 95,530 | 21,658 |
| Office Products | 62,585 | 14,947 |
| Total | 816,112 | 134,288 |

The main preparation selects users with 5–200 interactions, samples 100,000
eligible users with seed 42, and iteratively filters to at least five interactions
per user and two per item. The retained snapshot has **80,471 users**, **129,644
warm items**, and **655,170 training interactions**. The median full sequence
length is seven; the median item has two training interactions. Source and data
hashes are in [baseline_evidence.json](../../../artifacts/main/derived/baseline_evidence.json).

```text
Hypothetical broader history: headphones → novel → phone case → shoes → printer
Included-category history:   headphones ───────→ phone case ─────────→ printer

Retained user sequence:      A → B → C → D → V → T
Split:                      train history   valid test
Training pairs:             [A]→B, [A,B]→C, [A,B,C]→D
Validation:                 [A,B,C,D]→V
Test:                       [A,B,C,D,V]→T
```

These are rating records treated as implicit interactions, not complete browsing
or purchase histories. “Next” means next **retained** event; category and k-core
filtering can remove intervening events. The split is chronological per user,
not a shared calendar cutoff across users. “All catalog” means the retained
134,288 items, not every product in Amazon.

Popularity uses training interactions: tail 1–5, mid 6–20, head >20, unseen 0.
Evaluation excludes already-seen items. Warm validation has 75,989 targets; warm
test has 74,179. The remaining test targets are interaction-unseen. Content
features exist for these items, but availability of a representation does not
establish effective cold-start ranking. See the fuller
[evaluation contract](../research_log.md#evaluation-contract).

## B01 — SASRec's evaluation NaNs

**Symptom, recorded.** Training loss fell from about 1.26 to 0.24 while validation
Recall@20 stayed near 0.0001–0.0002. The NumPy non-writable-array warning appeared
in the same log but was not the cause of the ranking failure.

**Mechanism.** With left padding and causality, padded queries had no allowed
keys. In the locally tested PyTorch 2.14 execution paths, training handled the
empty attention rows with finite outputs, while evaluation's fast path produced
NaNs. Later layers projected padded NaNs into keys and values. Zero attention
weight does not sanitize a NaN value: `0 × NaN = NaN`. The contamination reached
real item states and their catalog scores; `topk` still returned indices.

| Query / key | PAD0 | PAD1 | A | B |
|---|:---:|:---:|:---:|:---:|
| PAD0, before fix | × | × | × | × |
| PAD1, before fix | × | × | × | × |
| A | × | × | ✓ | × |
| B | × | × | ✓ | ✓ |

**Fix, implemented.** Give each padded query its own diagonal self-connection.
Real queries still exclude padding and future keys. Reject non-finite scores
before ranking. Copy candidate IDs into a tensor to remove the unrelated array
warning. See [attention masking](../../../src/models/sasrec.py) and
[regression tests](../../../tests/test_sasrec.py).

**Recorded diagnostics.** 79,819 of 80,471 validation histories required padding.
In the first 256-user sample, 254 score rows were non-finite before the fix and
zero afterward. The diagnostic checkpoint reported epoch 9, which is not the
same run identity as every pasted training log. With the same weights, disabling
the problematic fast path and then applying the mask fix gave identical aggregate
results in the recorded diagnostic:

| Warm-target metric | Corrected saved SASRec | Popularity on the same split |
|---|---:|---:|
| Validation Recall@20 | 0.021529 | 0.022951 |
| Validation NDCG@20 | 0.008548 | 0.009912 |
| Test Recall@20 | 0.018981 | 0.018819 |
| Test NDCG@20 | 0.007384 | 0.008119 |

The checkpoint had recorded validation recall 0.000184: the diagnostic recovered
about **117×** higher recall without retraining. The raw diagnostic output and
original checkpoint identity are not preserved as immutable experiment artifacts;
this table is discussion evidence, not a new September 19 evaluation. Reused
`sasrec_best.pt` must not be assumed to contain those same weights today.

**Interpretation and decision.** The initial metric collapse was an evaluation
correctness problem. Falling sampled loss did not contradict it. Checkpoint
selection based on the broken metrics was unreliable, so subsequent experiments
needed a fresh corrected baseline. This fix did not demonstrate that SASRec
decisively beats popularity on ranking quality.

## B02 — Timestamp ties and two-tower startup errors

**Recorded failure.** `python -m scripts.train_two_tower --smoke --epochs 1`
stopped at the assertion requiring `[train..., valid, test]`.

For user **25287**, positions 17 and 18 shared timestamp **1456865140000**.
Training item **14709** preceded validation item **7131** in the assigned split.
Sorting by `(timestamp, item_idx)` reversed those events. This could make
`sequence[:-2]` include a held-out item or exclude a genuine training item.

**Fix.** Build sequences by `(user_idx, position)`, the order used to assign the
split. Rebuild `user_sequences.parquet`, and use that order in SASRec evaluation.
Six users' item orders changed; only one violated the split-order assertion.
The saved sequences were rechecked during this documentation pass and match the
original positions, including aligned ratings, timestamps, and split labels.

An initial loader-side reconstruction was deliberately removed after fixing the
preparation stage. [Two-Tower loading](../../../scripts/train_two_tower.py) reads the
prepared parquet and retains the assertion. The canonical construction belongs
in [prepare_model_data.py](../../../scripts/prepare_model_data.py).

Other startup blockers were fixed in the same debugging session: missing
`NextItemDataset` import, `self.init` instead of `nn.init`, undefined mapping hash,
and a loop expecting two logits tensors from a model returning a `[B, 65]` tensor.
Two-Tower now uses cross-entropy with the positive in column zero. These were
correctness repairs, not a controlled comparison of alternative losses.

## B03–B05 — Training coverage, capacity, and the 0.10 goal

### The training procedure changed over time

| Stage | Example and target definition | Objective |
|---|---|---|
| Initial SASRec in the discussion | One user window; next-item supervision at every valid position in the last `max_len + 1` training items | Binary logistic loss, one sampled negative per position |
| Initial Two-Tower | Last training prefix plus one random prefix per user each epoch | Cross-entropy over one positive and 64 negatives after the startup fix |
| Current SASRec | Every nonempty training prefix as an example; latest `max_len` context items | Binary logistic loss with mean loss across 64 negatives |
| Current Two-Tower | Same all-prefix coverage, scoring one positive and 64 negatives | Cross-entropy over the 65 candidate logits |

The early phrase “as SASRec already does” referred to per-position supervision;
the initial SASRec still truncated long histories to one training window. It was
not identical to enumerating every prefix across a long history. Current code
for both datasets enumerates all prefixes. Never infer the loss of an older run
solely from today's implementation.

All-prefix Two-Tower increased examples from **160,942 to 574,699** per epoch,
about **3.57×**. At batch 1,024, this is 562 batches; at 2,048, 281. Cumulative
offsets identify `(user, target position)` without materializing copied prefixes.
Negative samples change with epoch while contexts and targets remain fixed.
Negatives exclude the full user's training history; held-out events are not used
to construct that exclusion set. Smoke mode caps training at 2,048 examples and
each evaluation at 512 users.

### What the capacity suggestions actually established

The first suggestion was a smaller model because histories and item counts were
sparse. The user challenged that inference. Sparse data and decreasing training
loss alone did not establish an overcapacity bottleneck, especially before
evaluation was repaired. Later width runs provide mixed empirical evidence, not
a universal rule that smaller or larger models win.

Increasing `d_model` expands representation and projection capacity. At fixed
width, changing `n_heads` partitions that width rather than multiplying it;
`head_dim = d_model / n_heads`. More layers add processing depth, not additional
accessible history beyond `max_len`. The suggested isolated tests were width
128→256, heads 4→8, and depth 2→3. The retained runs include width and depth
changes; no attributable fixed-width head-count sweep was recovered.

The saved Two-Tower 128-d, 2-layer all-prefix run has valid warm R@20 **0.037466**
and test warm R@20 **0.030925**. The earlier 2-layer run has **0.034150** and
**0.030265**, respectively. Training examples, batch size (1,024→2,048), and
epoch budget (20→50) changed; the later checkpoint was selected at epoch 28.
This is a useful combined result, not an isolated causal gain from all prefixes.

### Why aggregate recall remained low

Measured from the [retained earlier Two-Tower history](../../../artifacts/main/training/two_tower_history_20260908T124156726516Z.json)
and the [later all-prefix history](../../../artifacts/main/training/two_tower_history_20260908T125631038709Z.json):

| Warm target group | Targets | Share | Earlier R@20 | Later all-prefix R@20 |
|---|---:|---:|---:|---:|
| Tail | 33,742 | 45.5% | 0.001837 | 0.002638 |
| Mid | 19,141 | 25.8% | 0.003082 | 0.005224 |
| Head | 21,296 | 28.7% | 0.099737 | 0.098845 |

The earlier discussion quoted a nearby standalone snapshot (warm R@20 0.030292,
head 0.099784, mid 0.003135). That filename was subsequently overwritten. The
table above uses surviving timestamped histories and does not silently equate
the slightly different snapshots.

The persistent weakness is rare-item retrieval: tail and mid together account
for 71.3% of warm targets. Head recall near 0.10 does not mean overall recall is
near 0.10. Improving content discrimination, training evidence, or ranking may
help; this distribution alone does not isolate their relative causes.

### Open proposals and evidence boundaries

| Proposal | Status and test needed |
|---|---|
| 256 or 1,024 negatives instead of 64 | No isolated sweep recovered; hold dataset, architecture, and objective fixed first |
| gSASRec / gBCE | Proposed for the original single-negative SASRec; not the current Two-Tower CE implementation |
| Full-catalog cross-entropy | Archived RecBole result exists, but multiple implementation choices differ; see [archive](../archive.md#recbole-and-saved-fullsubset-comparisons) |
| Projected text + learned item residual | Proposed for Two-Tower; keep unseen-item residual zero and regularize it; later GenRec variant D is a different experiment |
| More users' eligible training events for existing items | Proposed; preserve evaluation population and define temporal eligibility before adding data |
| Higher item-frequency threshold | Changes the population and problem difficulty; cannot establish improvement on the original benchmark |
| Longer training | Some completed runs select epochs 28/35 beyond the original limit of 20; does not establish that every model needs longer training |
| Stronger text model / longer text | Proposed, with later details in [H02](../research_log.md#2-baselines-implementation-and-hyperparameters--h01h03) |

Background motivating loss and embedding proposals: [gSASRec](https://arxiv.org/abs/2308.07192),
[Turning Dross Into Gold Loss](https://arxiv.org/abs/2309.07602), and
[content-based initialization with learned deltas](https://arxiv.org/abs/2507.19473).
Their reported gains are external evidence, not measured gains for this project.

## B08 — GPU throughput

**Recorded setting.** RTX 3070 Ti, 8 GiB VRAM, eight logical CPUs. The user observed
30–40% GPU utilization and roughly 30% VRAM occupancy. The original dataset ran
NumPy RNG creation, rejection sampling, and tensor construction synchronously
with `num_workers=0`; every training batch also called `loss.item()`.

A short loader-only benchmark used sampled actual prefixes, batch 1,024, four
warm-up batches and twenty measured batches, with startup excluded and worker
processes using `spawn`. Results retained in the discussion:

| Workers | Examples/s | Milliseconds/batch |
|---:|---:|---:|
| 0 | 12,097 | 84.7 |
| 2 | 18,755 | 54.6 |
| 4 | 28,751 | 35.6 |

This measured CPU batch preparation, not model throughput. The benchmark did not
establish a 2.38× end-to-end training speedup.

**Interventions.** Four loader workers, prefetch factor two, pinned memory, and
`spawn`; keep workers nonpersistent so `set_epoch` reaches the next epoch's
dataset copies. Also propose accumulating detached loss on the GPU and converting
to a Python scalar once per epoch, rather than synchronizing every batch.

**Recorded result.** After making both changes, the user reported **99% GPU
utilization and 45% VRAM occupancy**. No matched before/after examples-per-second
or epoch-time measurement was retained. The report supports reduced idle time,
not proof of fully saturated compute or a precise speedup. GPU utilization
measures time with kernels active; VRAM occupancy measures allocated capacity.
Filling VRAM is not an optimization objective. See [NVIDIA's definitions](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

**Current-code reconciliation.** On September 19, Two-Tower has the four-worker
loader settings, but `train_one_epoch` still contains `total_loss += loss.item()`.
The no-per-batch-synchronization change is therefore a historical report/proposal,
not a current-code claim. This documentation does not silently reapply it.

Batch 2,048 was subsequently used in recorded quality runs. Larger batches reduce
updates per epoch; compare examples/s and validation quality per elapsed training
time. Epoch count alone is not a matched compute budget. `torch.compile` was a
possible later optimization, not a recorded baseline speed experiment.

## B09 — Run history and plotting

Both runners write a unique `artifacts/main/training/<model>_history_<UTC timestamp>.json`.
The snapshot is atomically replaced after each completed epoch. It includes:

- Config, device, run ID, and start/finish timestamps.
- Training loss, learning rate, epoch duration, validation metrics by group, and
  whether that epoch became the best checkpoint.
- Selection metric, best epoch/value, early-stop flag, and final test metrics.
- Two-Tower also records smoke status, dataset/example counts, and mapping hash.

Two-Tower final metrics are nested under `test_metrics.warm_catalog` and
`test_metrics.all_catalog`. SASRec uses `test_metrics.<group>` in these histories.
Epoch loss is the runner's arithmetic mean of batch losses; it is not validation
loss, nor an exactly example-weighted epoch loss when the final batch is smaller.
Duration includes validation and checkpoint work, so it is not pure GPU training
time. Missing `finished_at` means incomplete recording; the cause is not known
from the JSON alone. `is_best` marks a record-setting epoch, not only the final
selected epoch.

Training logs link their best checkpoint to the history file, but ordinary
`*_best.pt` paths and `two_tower_metrics.json` are reused across runs. Timestamped
history preserves metrics, not every historical model's weights. Exact reruns
also need immutable checkpoints, code revision/diff, and data identities.

For plots, read `epochs[].epoch`, `epochs[].train_loss`, and
`epochs[].valid_metrics.warm.recall@20` / `ndcg@20`. Mark `best_epoch`, label metric
units, and distinguish smoke, incomplete, and completed runs. The JSON below
also exposes the selected-epoch metrics and full final metric dictionaries for
all retained baseline runs: [baseline_evidence.json](../../../artifacts/main/derived/baseline_evidence.json).

## Retained run ledger

The following tables are extracted from the timestamped histories. Values are
fractions. Missing strategy fields remain **not recorded**; the earlier procedure
described above comes from discussion, not an invented JSON field. Smoke rows
use only a subset and must not be ranked against full runs. The older SASRec
histories do not independently fingerprint the candidate catalog; warm scoring
is supported by the recorded runner and discussion, not immutable run revisions.

### SASRec

| Run (UTC on Sep 8) | d / heads / layers / max_len | Batch | Strategy | Epochs recorded / budget | State | Best epoch | Valid warm R@20 | Test warm R@20 |
|---|---|---:|---|---|---|---:|---:|---:|
| [08:51:59.839650](../../../artifacts/main/training/sasrec_history_20260908T085159839650Z.json) | 64 / 8 / 4 / 50 | 1024 | not recorded | 12 / 20 | complete | 9 | 0.027767 | 0.025007 |
| [08:56:55.994343](../../../artifacts/main/training/sasrec_history_20260908T085655994343Z.json) | 128 / 8 / 4 / 50 | 1024 | not recorded | 9 / 20 | complete | 6 | 0.025622 | 0.023308 |
| [09:01:54.248919](../../../artifacts/main/training/sasrec_history_20260908T090154248919Z.json) | 256 / 8 / 4 / 50 | 1024 | not recorded | 9 / 20 | complete | 6 | 0.025333 | 0.023093 |
| [13:50:21.287418](../../../artifacts/main/training/sasrec_history_20260908T135021287418Z.json) | 64 / 4 / 2 / 400 | 2048 | not recorded | 0 / 50 | incomplete | — | — | — |
| [13:50:38.969390](../../../artifacts/main/training/sasrec_history_20260908T135038969390Z.json) | 64 / 4 / 2 / 400 | 1024 | not recorded | 0 / 50 | incomplete | — | — | — |
| [13:50:49.322512](../../../artifacts/main/training/sasrec_history_20260908T135049322512Z.json) | 64 / 4 / 2 / 200 | 1024 | not recorded | 0 / 50 | incomplete | — | — | — |
| [13:52:50.796248](../../../artifacts/main/training/sasrec_history_20260908T135250796248Z.json) | 64 / 4 / 2 / 50 | 1024 | not recorded | 0 / 50 | incomplete | — | — | — |
| [14:31:26.184389](../../../artifacts/main/training/sasrec_history_20260908T143126184389Z.json) | 64 / 4 / 2 / 200 | 1024 | all_prefixes | 6 / 50 | complete | 3 | 0.030254 | 0.027595 |
| [14:43:22.654352](../../../artifacts/main/training/sasrec_history_20260908T144322654352Z.json) | 64 / 4 / 3 / 50 | 1024 | all_prefixes | 7 / 50 | complete | 4 | 0.030689 | 0.027555 |

### Content Two-Tower

| Run (UTC on Sep 8) | d / heads / layers / max_len | Batch | Strategy | Epochs recorded / budget | State | Best epoch | Valid warm R@20 | Test warm R@20 |
|---|---|---:|---|---|---|---:|---:|---:|
| [12:28:22.582355](../../../artifacts/main/training/two_tower_history_20260908T122822582355Z.json) | 128 / 4 / 2 / 50 | 1024 | not recorded | 1 / 1 | smoke | 1 | 0.002058 | 0.004228 |
| [12:41:56.726516](../../../artifacts/main/training/two_tower_history_20260908T124156726516Z.json) | 128 / 4 / 2 / 50 | 1024 | not recorded | 20 / 20 | complete | 19 | 0.034150 | 0.030265 |
| [12:50:21.681353](../../../artifacts/main/training/two_tower_history_20260908T125021681353Z.json) | 128 / 4 / 2 / 50 | 1024 | all_prefixes | 1 / 1 | smoke | 1 | 0.002058 | 0.000000 |
| [12:51:26.375320](../../../artifacts/main/training/two_tower_history_20260908T125126375320Z.json) | 128 / 4 / 2 / 50 | 1024 | all_prefixes | 2 / 50 | incomplete | 2 | 0.028570 | — |
| [12:56:31.038709](../../../artifacts/main/training/two_tower_history_20260908T125631038709Z.json) | 128 / 4 / 2 / 50 | 2048 | all_prefixes | 31 / 50 | complete | 28 | 0.037466 | 0.030925 |
| [13:10:00.075748](../../../artifacts/main/training/two_tower_history_20260908T131000075748Z.json) | 256 / 4 / 2 / 50 | 2048 | all_prefixes | 7 / 50 | incomplete | 7 | 0.023016 | — |
| [13:14:58.161044](../../../artifacts/main/training/two_tower_history_20260908T131458161044Z.json) | 64 / 4 / 2 / 50 | 2048 | all_prefixes | 9 / 50 | complete | 6 | 0.032189 | 0.027663 |
| [13:19:04.621494](../../../artifacts/main/training/two_tower_history_20260908T131904621494Z.json) | 128 / 4 / 3 / 50 | 2048 | all_prefixes | 38 / 50 | complete | 35 | 0.034558 | 0.029914 |

### Decisions from the ledger

- The completed 128-d, two-layer, batch-2,048 Two-Tower run wins its recorded
  validation selection metric among these baselines. It is the comparison point
  for future baseline changes, not a guarantee of the highest future test score.
- The 256-d Two-Tower run has only seven recorded epochs and no final test result;
  its lower partial score does not establish that greater width cannot work.
- The three-layer run's validation selection score is lower than the two-layer
  run's. Its epoch-35 checkpoint was nevertheless the exported teacher for later
  SID research, according to [export metadata](../../../data/processed/item_two_tower_embeddings.json).
- There is no completed isolated head-count sweep, no repeated-seed confirmation,
  and no successful baseline Recall@20 ≥ 0.10 result in the retained evidence.
- The later SID and decoding hypotheses have separate tests and outcomes in
  [the research log](../research_log.md); they should not overwrite this baseline
  history or be presented as matched baseline ablations.

Use the [experiment template](../protocols/experiment_template.md) to state a falsifiable
prediction and the evaluation contract before the next run.
