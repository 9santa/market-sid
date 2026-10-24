# MovieLens 1M GenRec benchmark

Run from the repository root. Data and weights stay in this benchmark directory; results go to `artifacts/benchmarks/movielens-1m/`. Generated files are Git-ignored. `--history-dir` can be set separately from `--checkpoint-dir`.

## Pipeline

```bash
python benchmarks/movielens-1m/preprocess.py
python benchmarks/movielens-1m/prepare_sids.py
python benchmarks/movielens-1m/train.py --variant B
python benchmarks/movielens-1m/evaluate.py --split test --beam-size 200 --batch-size 32
```

`preprocess.py` downloads [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/). To use existing files, pass `--raw-dir /path/to/ml-1m` with `ratings.dat` and `movies.dat`.

`prepare_sids.py` encodes titles, years, and genres with the pinned BGE-small model, trains RQ-VAE, and adds a collision token for unique SIDs. Cache metadata records the input and encoder settings. The RQ initialization uses full codebook sizes, including unused entries.

`train.py` uses the shared GenRec dataset, model, and validation code. JSON histories include loss, token accuracy, learning rate, epoch duration, and best epoch. Early stopping and checkpoint choice use **validation token loss**.

`evaluate.py` runs constrained beam search and Recall/NDCG@10/20/50. It restores the checkpoint variant and checks catalog, sequence, and SID identities.

## Protocol

- All ratings count as implicit interactions by default. `--min-rating 4`
  creates a different, positive-ratings-only benchmark.
- Duplicate user/movie pairs are removed. Users need at least four retained
  events: two training events, one validation event and one test event.
- Each user's events are sorted by timestamp, with movie ID breaking ties.
  The penultimate event is validation and the final event is test. These are
  rating timestamps, not known viewing times. This is a per-user split, not a
  global temporal cutoff.
- Every prefix within the training portion becomes a next-item example.
  Validation histories contain training events; test histories also include
  the validation event. The model receives the last `max_len` history items.
- The catalog contains all movies appearing in retained ratings, including
  held-out movies. IDs start at 1; 0 is PAD. Metadata-only embeddings and RQ-VAE
  use this full catalog, without held-out interaction labels.
- Evaluation searches the full retained catalog, with no sampled negatives.
  It removes all seen history items, then takes up to 50 recommendations.
  Beam search is approximate; the script reports users with fewer than 50 results.
- Head/mid/tail/unseen use the existing repository thresholds on training
  interaction counts: >20, 6–20, 1–5, and 0. Most MovieLens targets may be head
  under these thresholds. Groups with no targets are omitted.

MovieLens 1M contains 6,040 users, 3,706 rated movies, and 1,000,209 interactions. Results with sampled negatives or different splits are not directly comparable.

## Variants and hyperparameters

| Variant | Embeddings | Semantic output weights |
|---|---|---|
| A | Learned from scratch | Separate |
| B (default) | Initialized from RQ codebooks | Separate |
| C | Initialized from RQ codebooks | Tied |
| D | Frozen RQ base + trainable delta | Tied |

For C/D evaluation, select the corresponding checkpoint explicitly:

```bash
python benchmarks/movielens-1m/train.py --variant C --batch-size 512 --lr 0.0003 --dropout 0.1
python benchmarks/movielens-1m/evaluate.py --checkpoint benchmarks/movielens-1m/checkpoints/genrec_C_best.pt
```

Training exposes `--max-len`, `--d-model`, `--n-heads`, `--n-encoder-layers`,
`--n-decoder-layers`, `--dropout`, `--batch-size`, `--lr`, `--weight-decay`,
`--epochs`, `--patience`, `--seed`, and `--anchor-lambda` (D only).

SID preparation exposes `--encoder`, `--revision`, `--max-tokens`,
`--encode-batch-size`, `--epochs`, `--batch-size`, `--lr`, `--latent-dim`,
`--num-codebooks`, `--codebook-size`, and `--seed`.
For B/C/D, `--d-model` must match the RQ-VAE `--latent-dim`.
Changing the encoder or quantizer requires regenerating SIDs and retraining GenRec.
MovieLens supplies short titles/genres, not long plot descriptions.

All model scripts accept `--device cpu` or `--device cuda`.
Use `--data-dir` for separate preprocessing/tokenizer experiments and
`--checkpoint-dir` for separate training runs; `train.py --rq-checkpoint` selects
the tokenizer checkpoint. Best checkpoints are replaced on subsequent runs of
the same variant in the same directory; GenRec histories have unique timestamps.

## Smoke run

After preprocessing and SID preparation:

```bash
python benchmarks/movielens-1m/train.py --variant B --smoke
python benchmarks/movielens-1m/evaluate.py --checkpoint benchmarks/movielens-1m/checkpoints/genrec_B_smoke_best.pt --limit 10 --batch-size 10
```

This trains one epoch on at most 1,024 prefixes and validates on 32 users.
Smoke checkpoints have separate names and are not benchmark-quality models.
Use validation results for tuning, then evaluate the selected configuration on test.

## Target prefix survival

Add `--diagnose-prefixes` to evaluation to measure whether each target SID
prefix survives beam pruning at positions 1, 2, 3 and 4. This observes normal
generation without teacher forcing or changing beam selection.

```bash
python benchmarks/movielens-1m/evaluate.py --split test --beam-size 100 --batch-size 64 --diagnose-prefixes
```

The table reports all/head/mid/tail/unseen, group sizes, prefix survival rates
and item top-10 recall. Survival uses all targets in each group as its
denominator and is measured before seen-item filtering; top-10 uses the
filtered recommendations. Empty groups display N/A. JSON results include
`prefix_survival@1` through `prefix_survival@4` alongside the usual metrics,
and default output filenames include `_prefix_survival`. Omit `--limit`
to evaluate every target.
