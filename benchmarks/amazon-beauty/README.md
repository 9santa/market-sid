# Amazon Reviews 2023: Beauty and Personal Care

GenRec benchmark using `Beauty_and_Personal_Care` from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), not `All_Beauty` or Amazon 2014.

From the repository root:

```bash
python benchmarks/amazon-beauty/preprocess.py
python benchmarks/amazon-beauty/prepare_sids.py
python benchmarks/amazon-beauty/train.py --variant C
python benchmarks/amazon-beauty/evaluate.py --split test --beam-size 200 --batch-size 64
```

Data and weights stay here; metrics and histories go under `artifacts/benchmarks/amazon-beauty/`. Generated files are Git-ignored. Use `--history-dir` to change where histories go.

## Preprocessing

The pipeline uses the same stages as the other Amazon benchmarks:

- Download the official **5-core rating-only** CSV. All rating values count as
  implicit interactions; review text is not used.
- Keep the earliest review per user/item pair, preventing repeated items from
  becoming seen validation/test targets. Timestamp ties are broken by `parent_asin`.
- Keep users with 5–200 distinct items and sample up to 100,000 users with seed 42.
- Apply the existing iterative k-core: at least 5 events per user and 2 per item.
- Build item text with the shared Amazon cleaner: title, brand/store, categories,
  features, description and details, with the existing 4,000-character limit.
- Remove products without usable metadata and repeat k-core if necessary.
- Sort each user's events chronologically. The penultimate event is validation;
  the final event is test. All earlier events are training data.
- Assign sorted item IDs starting at 1; 0 is reserved for PAD.

The split is chronological within each user, after k-core filtering; it is not a global calendar cutoff. The catalog retains held-out items. Head/mid/tail/unseen groups use training interactions only.

Metadata Parquet files are pinned to revision
`b7ac094bf775b1ae467e61a98d343207e826c4e6`, where this category is available in Parquet.
Ratings are pinned to `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`.
The filtering settings, revisions and resulting counts are saved in `preprocessing.json`.

Use all eligible users instead of sampling:

```bash
python benchmarks/amazon-beauty/preprocess.py --max-users 0
```

Other options: `--min-user-interactions`, `--max-user-interactions`,
`--min-item-interactions`, `--seed`, and `--data-dir`.
Existing downloads can be supplied with `--ratings-path /path/to/ratings.csv`
and `--metadata-dir /path/to/raw_meta_Beauty_and_Personal_Care`.

## Semantic IDs and training

`prepare_sids.py` encodes Beauty product text with the repository's pinned BGE-small
model, then trains a **new Beauty RQ-VAE from scratch**. It does not load MovieLens
or the original marketplace RQ checkpoint. Three code levels plus a collision suffix
give every item a unique SID.

The default tokenizer uses 30 epochs, three 256-entry codebooks and a 128-dimensional
latent space. Options include `--encoder`, `--revision`, `--max-tokens`,
`--encode-batch-size`, `--epochs`, `--batch-size`, `--lr`, `--latent-dim`,
`--num-codebooks`, `--codebook-size`, and `--seed`.

Training uses the existing GenRec model, prefix dataset, loss and validation functions.
The default is **C**: RQ initialization plus tied semantic input/output weights,
128 dimensions, 3 encoder layers, 2 decoder layers, batch size 512 and learning rate 3e-4.
Variants A/B/D are also available. For B/C/D, `--d-model` must match tokenizer `--latent-dim`.

Training exposes the same hyperparameters as the MovieLens benchmark and writes a
timestamped JSON history. Best checkpoint selection and early stopping use validation
token loss. Test events are excluded from training and validation.

Evaluation restores the checkpoint's variant and searches the full retained catalog
with constrained beam search. It filters the full seen history before taking the top
50 and reports Recall/NDCG at 10/20/50, including popularity groups. Test histories
include validation events. It reports any users receiving fewer than 50 recommendations.
Beam search is approximate; these are not sampled-negative metrics.

Checkpoint/data signatures prevent evaluation with different mappings, splits or SIDs.
For separate experiments use `--data-dir`, `--checkpoint-dir`, and `--rq-checkpoint`
when training; evaluation accepts `--checkpoint` and `--output`.
All model scripts support `--device cpu` or `--device cuda`.

## Smoke run

After preprocessing and SID preparation:

```bash
python benchmarks/amazon-beauty/train.py --variant C --smoke
python benchmarks/amazon-beauty/evaluate.py --checkpoint benchmarks/amazon-beauty/checkpoints/genrec_C_smoke_best.pt --limit 10 --batch-size 10
```

This trains one epoch on up to 1,024 prefixes and validates on 32 users. Smoke
checkpoints have separate names and should not be used to report benchmark quality.

## Target prefix survival

Add `--diagnose-prefixes` to evaluation to measure whether each target SID
prefix survives beam pruning at positions 1, 2, 3 and 4. This observes normal
generation without teacher forcing or changing beam selection.

```bash
python benchmarks/amazon-beauty/evaluate.py --split test --beam-size 100 --batch-size 64 --diagnose-prefixes
```

The table reports all/head/mid/tail/unseen, group sizes, prefix survival rates
and item top-10 recall. Survival uses all targets in each group as its
denominator and is measured before seen-item filtering; top-10 uses the
filtered recommendations. Empty groups display N/A. JSON results include
`prefix_survival@1` through `prefix_survival@4` alongside the usual metrics,
and default output filenames include `_prefix_survival`. Omit `--limit`
to evaluate every target.
