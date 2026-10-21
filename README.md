# MarketSID

**Work in progress.** This repository explores generative sequential recommendation
with semantic item IDs (SIDs), with a focus on marketplace catalogs, long-tail
products, and items without training interactions. It contains research pipelines
and experiments; the training setup and evaluation methods are still evolving.

The main question is whether product metadata can provide useful item addresses
for an autoregressive recommender, and how SID construction and decoding affect
retrieval across popularity groups.

For the history of the project, start with the **[research notebook](docs/experiments/README.md)**.
It documents hypotheses, completed and unsuccessful experiments, measured results,
changes in interpretation, and open questions, with links to the supporting artifacts.
The [baseline development chapter](docs/experiments/results/baseline_development.md) covers
SASRec and Two-Tower debugging, training coverage, throughput, and Recall@20 runs;
use the [experiment template](docs/experiments/protocols/experiment_template.md) for new trials.

## Approach

```text
Product metadata → BGE text embedding → RQ-VAE → semantic item ID
                                                       ↓
User history → Transformer encoder → autoregressive SID decoder
                                                       ↓
                                      Catalog-constrained beam search
                                                       ↓
                                         Filter seen items → recommendations
```

The residual-quantized VAE (RQ-VAE) turns a continuous item representation into
discrete codes. GenRec learns to predict the next item's codes from the user's
history. A trie restricts generated sequences to SIDs present in the catalog.

The repository also includes SASRec and a content-based Two-Tower model. The
Two-Tower item network transforms text embeddings into recommendation embeddings;
these can serve as tokenizer inputs or as a behavioral teacher during tokenizer
training.

## Setup

Run commands from the repository root. Use a Python virtual environment and a
PyTorch installation appropriate for your CPU or CUDA setup. The current local
environment uses Python 3.14; a fresh environment is not yet covered by a pinned
dependency manifest.

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch for your machine, then the remaining Python dependencies.
python -m pip install numpy polars scipy scikit-learn sentence-transformers \
  "datasets==3.6.0" huggingface-hub pyarrow matplotlib
```

The main preprocessing script uses the Hugging Face dataset-script loading API
with `trust_remote_code=True`; `datasets==3.6.0` is the version in the current
environment. The commands above are a setup starting point, not a fully locked
environment. Dataset and encoder downloads require network access. Training and
full-catalog evaluation are intended for a CUDA machine; scripts generally fall
back to CPU when CUDA is unavailable.

## Main dataset

The main pipeline combines three Amazon Reviews 2023 categories:

- Electronics
- Cell Phones and Accessories
- Office Products

It loads the 5-core rating-only data, filters and samples users, applies iterative
k-core filtering, and creates a chronological leave-two-out split per user:
the penultimate interaction is validation, the final interaction is test, and
earlier interactions form training histories. Ratings are treated as implicit
interactions. This is a per-user split, not a global temporal cutoff.

The current processed snapshot has **80,471 users**, **134,288 items**, and
**574,699 next-item training examples**. Regenerating data can change these counts.
Item indices start at 1; index 0 is reserved for padding.

Popularity groups use training interaction counts:

| Group | Training interactions per item |
|---|---:|
| Head | >20 |
| Mid | 6–20 |
| Tail | 1–5 |
| Unseen | 0 |

The tokenizer uses the retained catalog, including metadata for held-out items.
Here, **unseen means interaction-unseen during training**; it does not mean the
item's metadata was unavailable when constructing the tokenizer and catalog trie.

### Prepare data and text embeddings

```bash
python -m scripts.prepare_data
python -m scripts.prepare_metadata
python -m scripts.prepare_model_data
python -m scripts.encode_items
```

These stages save the indexed catalog, interactions, user sequences, and text
embeddings under `data/processed/`. Product text is assembled from metadata such
as titles, categories, features, and descriptions.

The default encoder is the pinned `BAAI/bge-small-en-v1.5` model, with normalized
**384-dimensional embeddings** and a **512-token maximum sequence length**.
Embedding cache metadata records the catalog/text identity and encoder settings.

### Content SID baseline

```bash
python -m scripts.train_rqvae
python -m scripts.export_semantic_ids
python -m scripts.train_genrec --variant C --sid-source content
```

This tokenizer uses three learned RQ codes and an additional collision suffix
to give each item a unique four-token SID. GenRec C initializes semantic embeddings
from the matching RQ codebooks and ties semantic input/output weights.

Start with a small validation evaluation:

```bash
python -m scripts.evaluate_genrec --sid-source content --split valid \
  --beam-size 100 --batch-size 10 --limit 10 --diagnose-prefixes
```

For a full evaluation, remove `--limit` and choose a batch size that fits GPU
memory. Use validation for experiment selection, then evaluate the selected
configuration on test:

```bash
python -m scripts.evaluate_genrec --sid-source content --split test \
  --beam-size 100 --batch-size 64 --diagnose-prefixes
```

Evaluation reports Recall and NDCG at 10, 20, and 50, including popularity groups.
It searches the retained catalog without sampled evaluation negatives. Beam
search is approximate; seen items are removed after generation, so some users
can receive fewer than 50 recommendations.

### Recommendation baselines and Two-Tower SIDs

```bash
python -m scripts.train_sasrec
python -m scripts.train_two_tower
```

These scripts train their respective models and run recommendation evaluation.
To construct SIDs from the trained Two-Tower item representation:

```bash
python -m scripts.export_two_tower_embeddings
python -m scripts.train_twotower_rqvae
python -m scripts.export_twotower_semantic_ids
python -m scripts.train_genrec --variant C --sid-source twotower
python -m scripts.evaluate_genrec --sid-source twotower --split valid \
  --beam-size 100 --batch-size 64 --diagnose-prefixes
```

The exported item vectors come from the content-based item network, so an item
does not need a learned ID embedding to obtain a representation.

## Model and tokenizer experiments

GenRec's `--variant` selects the embedding/output configuration:

| Variant | Semantic embeddings | Semantic output weights |
|---|---|---|
| A | Learned from scratch | Separate |
| B | Initialized from RQ codebooks | Separate |
| C | Initialized from RQ codebooks | Tied to input embeddings |
| D | Frozen RQ base + trainable, regularized delta | Tied to input embeddings |

Most main-dataset hyperparameters live in `CONFIG` in
[`scripts/train_genrec.py`](scripts/train_genrec.py): history length 50,
model dimension 128, 4 attention heads, 3 encoder layers, 2 decoder layers,
dropout 0.2, batch size 2048, and learning rate `3e-4`. Checkpoint selection and
early stopping use validation token loss. The main evaluator currently loads
**GenRec variant C** for the requested SID source.

Separately, `--sid-source` selects the tokenizer:

| SID source | Tokenizer input | Additional supervision | Structure |
|---|---|---|---|
| `content` | BGE | None | 3 RQ + collision suffix |
| `twotower` | Two-Tower item vectors | None | 3 RQ + collision suffix |
| `rq4_a` | Two-Tower item vectors | None | 4 RQ |
| `rq4_b` | BGE | Predict Two-Tower vectors from the final RQ sum | 4 RQ |
| `rq4_c` | BGE | Predict Two-Tower vectors at each partial RQ depth | 4 RQ |

The four-RQ experiments enforce uniqueness during export by assigning distinct
fourth-level codewords within each three-code prefix bucket using minimum-cost
matching. There is no fifth collision token. A bucket must fit within the final
codebook's capacity. Assignments depend on the catalog, so changing the catalog
can change existing SIDs.

The tokenizer experiment letters A/B/C are independent of GenRec's A/B/C/D
variants. See [the SID ablation guide](docs/experiments/protocols/sid_ablation.md) for training,
export, and comparison commands.

Keep each SID file paired with its own RQ checkpoint. Main GenRec checkpoints
are saved as `checkpoints/genrec_<sid_source>_<variant>_best.pt`; rerunning the same
configuration replaces that best-checkpoint file. Training histories have
timestamped names. Changing SIDs requires retraining the corresponding GenRec.

## Diagnostics and current research

Experiments so far show a substantial head/mid/tail performance gap. Changing
the tokenizer has not consistently resolved it. Current work investigates where
targets disappear during generation and how training frequency affects SID-path
scores. These are ongoing experiments, not established improvements.

- **Prefix survival:** `--diagnose-prefixes` records whether the true target
  prefix survives each beam-pruning step, before seen-item filtering.
- **Path likelihood:** `--diagnose-paths` scores every true SID with teacher
  forcing and relates token/prefix probabilities to training frequencies.
  `--shuffle-histories` adds a comparison using another user's history.
- **Prior correction:** `--prior-alpha-sweep` evaluates decoding scores that
  subtract a weighted, smoothed conditional training log prior.
- **Trie normalization:** `--trie-normalize` normalizes probabilities over valid
  catalog children at each prefix, instead of the full token vocabulary.

For example, compare normal decoding with trie normalization on validation:

```bash
python -m scripts.evaluate_genrec --sid-source twotower --split valid \
  --beam-size 100 --batch-size 64 --diagnose-prefixes
python -m scripts.evaluate_genrec --sid-source twotower --split valid \
  --beam-size 100 --batch-size 64 --diagnose-prefixes --trie-normalize
```

Add `--limit 100` to both commands for a small run. See
[the path-likelihood and decoding guide](docs/experiments/protocols/sid_path_likelihood.md)
for definitions, output files, prior sweeps, and interpretation limits.
Results are saved under `artifacts/`.

## Separate benchmarks

For **ItemKNN, GRU4Rec, LightGCN, and an adapted TIGER baseline** on the existing
main dataset, use the [model comparison pipeline](benchmarks/main-comparison/README.md).
It provides training, evaluation, and a report alongside GenRec with matching
data, user populations, and candidate catalogs.

Each benchmark has its own preprocessing, tokenizer training, GenRec training,
and evaluation pipeline, with separate data and checkpoints. RQ-VAE codebooks
are trained on that benchmark's item representations.

- [Amazon Beauty and Personal Care](benchmarks/amazon-beauty/README.md)
- [Amazon Industrial and Scientific](benchmarks/amazon-industrial-scientific/README.md)
- [MovieLens 1M](benchmarks/movielens-1m/README.md)

Follow each benchmark's README for its exact protocol. In particular, the Amazon
benchmark preprocessing removes duplicate user/item pairs and products without
usable metadata; the main pipeline's filtering is not identical. Results should
not be compared across differing splits or candidate populations as a controlled
ablation.

## Repository layout

```text
src/data/       Datasets, preprocessing helpers, embedding cache
src/models/     Recommenders, RQ-VAE variants, trie, beam search
scripts/        Main-dataset preparation, training, evaluation, diagnostics
benchmarks/     Separate dataset pipelines
docs/           Documentation index and repository layout
docs/experiments/ Protocols, research log, reviews, and interpreted results
tests/          Existing unit tests
data/           Generated datasets and embeddings
checkpoints/    Main-dataset model weights
artifacts/main/ Training histories, evaluations, diagnostics, and reports
artifacts/benchmarks/ Results grouped by benchmark dataset
artifacts/archive/ Historical runs and source snapshots
```

See the [documentation index](docs/README.md) and [output layout](docs/repository_layout.md)
for where to find each experiment. Benchmark weights remain in
`benchmarks/<dataset>/checkpoints/`; their results live under `artifacts/benchmarks/`.
Generated data, weights, and raw outputs are ignored by Git. Small research evidence
summaries and three previously tracked baseline histories are retained.
Environment packaging, broader validation, and final benchmark reporting remain unfinished.

Run the existing lightweight tests with:

```bash
python -m unittest discover -s tests
```
