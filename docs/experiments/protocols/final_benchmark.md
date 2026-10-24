# Final comparable benchmark: new Amazon user cohort

The original [machine-readable protocol](../../../benchmarks/main-comparison/final_protocol.json)
is already recorded in prepared data and training runs. Preserve its contents:
its SHA-256 identifies the dataset, checkpoints, selection manifest, and evaluations.

**Current extension: GenRec A/B/C/D.** Follow the
[expanded protocol commands](genrec_all_variants.md) to add the other variants,
reuse compatible C/baseline runs, and freeze/evaluate all **40 checkpoints**.
The commands below document the original 22-checkpoint benchmark and preparation.
Do not use its 22-checkpoint freeze/evaluation commands for the expanded comparison.

## Question and scope

Primary comparison: **GenRec C with newly trained Two-Tower SIDs versus Content
Two-Tower**, by full-catalog **Recall@50**. This compares complete retrieval
pipelines, not isolated architectures: the GenRec pipeline also trains a teacher
and tokenizer. Report that cost. BGE-SID GenRec C versus adapted TIGER with the
same BGE SIDs is a secondary tokenizer-controlled comparison. Include SASRec,
GRU4Rec, LightGCN, and ItemKNN.

The rare-item hypothesis is evaluated through separately reported mid/tail/unseen
Recall@50, counts, and paired uncertainty. Choosing a checkpoint by overall recall
can favor head items; therefore an overall win does not establish a rare-item win.
Do not switch the selection metric to a subgroup score after seeing the outcomes.
No new SID variants, hybrid quotas, prior correction, or trie normalization in
this benchmark. Earlier results remain exploratory and are not mixed into its tables.

## New distribution, not a shuffled old test

Use a fresh sample from the same pinned Amazon 2023 categories: Electronics,
Cell Phones and Accessories, Office Products. Exclude **all user IDs** in the old
main user mapping and locally present Amazon benchmark sequence exports, plus any
additional previously used cohorts supplied with `--exclude-users`.

Preparation uses seed **20261021**, at most 100,000 eligible users, earliest review
per user/item, 5–200 distinct items/user, iterative user/item support filtering,
metadata availability, and stable timestamp/item-ID ordering. After filtering,
the last two events of every user become validation and test. Earlier events form
training. Item IDs are rebuilt, PAD is zero, and the entire new catalog is shared.
Groups use **training** counts only: unseen=0, tail=1–5, mid=6–20, head>20.
Deduplication makes held-out targets absent from their own earlier histories.

This is a new evaluation cohort, not an independent new dataset or prospective
experiment: categories/items may overlap the old study, pretrained BGE remains
shared, metadata describe a retrospective catalog, and the source is already
5-core filtered. Eligibility filtering also uses complete observed sequences.
The split is chronological **within users**, not across the global timeline.
Training can include another user's events later than a target's timestamp.
Consequently results support a known-catalog offline next-item comparison, not a
claim about future catalog availability or an online deployment.

A random re-split of previously examined interactions cannot erase research
adaptation to that data. A strict future-time claim would need a separate global
cutoff design and time-dependent candidate catalog. See the primary studies on
[temporal leakage](https://arxiv.org/abs/2010.11060) and
[sequential splitting protocols](https://arxiv.org/abs/2507.16289).

## Shared training and selection contract

| Setting | Fixed rule |
|---|---|
| Checkpoint selection | Maximize **validation all-target Recall@50** |
| Validation | Full validation population after every epoch; full catalog |
| Patience | 5 consecutive evaluations without strict improvement |
| Tie handling | Keep earlier epoch |
| Epoch cap | 50 for all neural recommenders |
| Repetitions | Training seeds 42, 43, 44; keep every seed, never choose the best seed |
| ItemKNN | Deterministic fit, one run at seed 42; no early stopping |
| Hyperparameter search | One configuration/model, frozen before new results; no new search |
| Histories | Last 50 items for sequence models; full seen history for filtering |
| Generative retrieval | Beam 100; original full-vocabulary probabilities; no prior correction |
| Other retrieval | Exact full-catalog scores; PAD and seen items excluded |
| Final refit | None; evaluate selected train-only checkpoint, no train+valid refitting |

The objective functions remain appropriate to each model: token CE for GenRec/
TIGER, item CE for GRU4Rec, sampled BCE for SASRec, sampled-softmax CE for
Two-Tower (64 warm negatives), BPR for LightGCN, and cosine co-occurrence for
ItemKNN. They share **selection**, not a common loss scale. The repository's
existing all-prefix training is retained; LightGCN fits all train graph edges.

Model-specific dimensions, learning rates, batch sizes, dropout and precision are
in the JSON. GenRec/SASRec/Two-Tower retain the main trainers' architectural
settings; comparison baselines use the previously documented starting settings.
This is a fixed-budget comparison informed by earlier development, **not equal
historical tuning effort or a claim that every architecture has been optimized**.
Epochs are not equal FLOPs. Record epochs, duration and parameter counts.

Changing batch size or other training settings after an OOM requires a protocol
revision before inspecting test. Evaluation batch/decode chunk sizes can change
without changing the ranking definition. CUDA training is not forced bitwise
deterministic; three seeds measure some training variation, not all uncertainty.

RecBole likewise distinguishes training loss from its
[validation metric for early stopping](https://www.recbole.io/docs/user_guide/config/evaluation_settings.html).
The legacy standalone trainers keep their old behavior; use the common runner
below for every final model.

## Features and tokenizer isolation

All content models share pinned BGE-small, 512 tokens, normalized 384-dimensional
features for this ordered catalog. GenRec content and TIGER share the exact same
three-RQ-plus-collision SIDs. GenRec C uses RQ initialization and semantic tying.

Train the Two-Tower teacher from scratch on the **new train split**, select it
by the same validation Recall@50, and use the **predeclared seed 42** checkpoint.
Export catalog representations, train a fresh RQ-VAE, and hold those SIDs fixed
across all three GenRec seeds. Never choose a teacher seed by test performance.
This teacher/tokenizer is a fixed preprocessing component: the three GenRec seeds
do not measure tokenizer or teacher-seed variance.

RQ-VAE is preprocessing, not an interaction recommender: 30 fixed epochs, seed 42,
select lowest catalog reconstruction loss, with no held-out interaction labels.
It sees the catalog metadata of interaction-unseen items; this is explicitly a
**known-catalog, interaction-unseen** setting, not arrival of new items after deployment.
Old behavioral teachers, old codebooks and old recommender checkpoints are not reused.

## Commands

Run from the repository root, with the existing environment. These are intentionally
manual stages: preparation and training are substantial jobs. No new dependencies.
Use `python` from the activated `.venv`. Each run refuses an existing checkpoint
folder. The final outputs remain ignored by Git.

```bash
BENCH=benchmarks/main-comparison
PROTOCOL=$BENCH/final_protocol.json
FINAL_DATA=data/final-v1
FINAL_CKPT=$BENCH/checkpoints/final-v1
FINAL_ART=artifacts/benchmarks/final-main

python "$BENCH/prepare_final.py" --protocol "$PROTOCOL" --output-dir "$FINAL_DATA"

# Text features and content SID tokenizer, built only on the new catalog.
python "$BENCH/prepare_features.py" --protocol "$PROTOCOL" --data-dir "$FINAL_DATA" \
  --checkpoint-dir "$FINAL_CKPT/tokenizers" --history-dir "$FINAL_ART/training/tokenizers"
```

Existing rating-only CSV/metadata downloads can be supplied through `--ratings-dir`
and `--metadata-root`; see `prepare_final.py --help`. Keep the generated
`benchmark_manifest.json`. The script prints population sizes, not recommendation
quality. Do not keep changing cohort seeds to obtain favorable results.

Train the nongenerative models. `--protocol` supplies the frozen model-specific
hyperparameters and rejects conflicting training overrides:

```bash
python "$BENCH/train.py" --protocol "$PROTOCOL" --data-dir "$FINAL_DATA" \
  --checkpoint-dir "$FINAL_CKPT" --history-dir "$FINAL_ART/training" \
  --model itemknn --run-name itemknn_s42 --seed 42

for model in gru4rec lightgcn sasrec two_tower; do
  for seed in 42 43 44; do
    python "$BENCH/train.py" --protocol "$PROTOCOL" --data-dir "$FINAL_DATA" \
      --checkpoint-dir "$FINAL_CKPT" --history-dir "$FINAL_ART/training" \
      --model "$model" --run-name "${model}_s${seed}" --seed "$seed" --device cuda || break 2
  done
done

python "$BENCH/prepare_features.py" --protocol "$PROTOCOL" --data-dir "$FINAL_DATA" \
  --checkpoint-dir "$FINAL_CKPT/tokenizers" --history-dir "$FINAL_ART/training/tokenizers" \
  --teacher-checkpoint "$FINAL_CKPT/two_tower_s42/best.pt"
```

Train GenRec C for both SID sources and TIGER for the content source:

```bash
for seed in 42 43 44; do
  for source in content twotower; do
    python "$BENCH/train.py" --protocol "$PROTOCOL" --data-dir "$FINAL_DATA" \
      --checkpoint-dir "$FINAL_CKPT" --history-dir "$FINAL_ART/training" \
      --model genrec --sid-source "$source" --run-name "genrec_${source}_s${seed}" \
      --seed "$seed" --rq-checkpoint "$FINAL_CKPT/tokenizers/rqvae_${source}_best.pt" \
      --device cuda --eval-batch-size 16 --decode-batch-size 128 || break 2
  done
  python "$BENCH/train.py" --protocol "$PROTOCOL" --data-dir "$FINAL_DATA" \
    --checkpoint-dir "$FINAL_CKPT" --history-dir "$FINAL_ART/training" \
    --model tiger --sid-source content --run-name "tiger_content_s${seed}" --seed "$seed" \
    --device cuda --eval-batch-size 16 --decode-batch-size 128 || break
done
```

All runs must finish normally. Inspect training/validation only. If changes to
methods or hyperparameters are needed, document that decision before final test.
For this fixed-configuration protocol, there is one selected checkpoint per seed.
Freeze all **22 checkpoints** (7 neural conditions × 3 seeds + ItemKNN):

```bash
python "$BENCH/freeze_selection.py" "$FINAL_CKPT"/*/best.pt \
  --protocol "$PROTOCOL" --data-dir "$FINAL_DATA" --output "$FINAL_ART/selection.json"
```

Only then evaluate test. Test requires membership in the frozen manifest and
refuses to overwrite existing outputs. This is an operational guard, not a claim
that a local file is inaccessible to the researcher.

```bash
for checkpoint in "$FINAL_CKPT"/*/best.pt; do
  run_name=$(basename "$(dirname "$checkpoint")")
  python "$BENCH/evaluate.py" --checkpoint "$checkpoint" --protocol "$PROTOCOL" \
    --data-dir "$FINAL_DATA" --selection-file "$FINAL_ART/selection.json" \
    --split test --device cuda --beam-size 100 --batch-size 16 --decode-batch-size 128 \
    --output "$FINAL_ART/evaluation/${run_name}_test.json" || break
done

python "$BENCH/final_report.py" "$FINAL_ART"/evaluation/*_test.json \
  --selection-file "$FINAL_ART/selection.json" --output "$FINAL_ART/reports/comparison.md"
```

## Reporting and stopping

Report mean ± sample SD over the three seeds for Recall/NDCG@10/20/50 and every
popularity group, with its denominator. The report saves Markdown and JSON.
Keep the per-user NPZ files for paired comparisons.

For the registered primary Recall@50 comparison, average the paired seed results
per user and bootstrap users (5,000 resamples; seed 20261021). These intervals are
conditional on the chosen split and fitted models; seed SD is reported separately.
Do not interpret zero observed unseen hits as proof that the true hit rate is zero.
Subgroup comparisons are exploratory and not multiplicity-adjusted. ItemKNN's
paired comparison uses only seed 42 of the baseline because its single fit is deterministic.

A positive semantic-assignment control in the old experiments remains valid
exploratory evidence, but does not prove GenRec beats a content baseline on rare
items. Report whichever outcome occurs. After this table, close the benchmark;
new test-informed designs become a subsequent exploratory study rather than edits
to this confirmatory protocol. Beauty/Industrial/MovieLens old results stay in a
separate robustness appendix, not the fresh main benchmark table.
