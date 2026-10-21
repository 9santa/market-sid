# Main dataset: ItemKNN, GRU4Rec, LightGCN, and TIGER

Compare these models with GenRec on the existing main Amazon dataset. Run all
commands from the repository root, in the project's Python environment. No data
download or new split is needed. Training, evaluation, and ItemKNN fitting run
only when you execute the commands below.

## Implementations

| Model | Implementation | Training input / objective |
|---|---|---|
| ItemKNN | Blockwise binary cosine similarity, top-k per candidate item | Training-only user–item co-occurrence; no neural training |
| GRU4Rec | Installed RecBole GRU4Rec, right-padded sequences | Every training prefix, full-vocabulary cross-entropy |
| LightGCN | Installed RecBole LightGCN | Training-only binary graph; BPR on training events with one warm negative per event |
| TIGER | Local adaptation using a randomly initialized Hugging Face T5 | Every training prefix; predict four content SID tokens |

ItemKNN never builds a dense item-by-item or all-user prediction matrix. Its
similarity pruning follows the candidate-column convention used by RecBole;
cosine ties are resolved by item ID. `--shrink` adds a shrinkage term to the
product of item norms. History scores sum similarities over distinct seen items.

GRU4Rec and LightGCN reuse the model classes, not RecBole's data splitting or
training runner. LightGCN overrides only adjacency construction to avoid the
removed SciPy `dok_matrix._update` API, preserving symmetric degree normalization.
Negative sampling excludes all of a user's training items and uses no held-out
labels. Validation edges are never added to the LightGCN graph.

Dependencies are already installed in the reviewed workspace: RecBole 1.2.1,
Transformers 5.16.1, PyTorch, NumPy, SciPy, and Polars. Nothing is installed by these
scripts. Package versions are recorded in each checkpoint/history. T5 uses numeric
tokens directly: no pretrained language-model download or SentencePiece tokenizer.

References: [RecBole models](https://www.recbole.io/docs/user_guide/model_intro.html),
[LightGCN](https://arxiv.org/abs/2002.02126), and
[TIGER](https://arxiv.org/abs/2305.05065).

## Evaluation contract

- Keep `data/processed/catalog_model.parquet`, `user_sequences.parquet`, and
  `interactions_model.parquet` unchanged. Their hashes are checked at evaluation.
- Preserve indexed user/item IDs and the existing leave-two-out order. Training
  excludes validation/test targets; test histories include validation events.
- Evaluate all catalog items; remove PAD and the **complete** seen history.
  Report Recall/NDCG@10/20/50 for all/warm/head/mid/tail/unseen, using the same
  training-count thresholds as GenRec.
- ItemKNN and sequential models can consume the validation event at test time.
  LightGCN uses its train-only learned user vector; the validation event affects
  seen filtering, not its graph or user embedding. This is a model difference.
- GRU4Rec and TIGER use the last 50 history items by default. ItemKNN uses the
  full observed history; LightGCN builds its graph from all training events.
- The three nongenerative models use exact full-catalog scoring. TIGER and
  GenRec use catalog-constrained beam search; matching beam width does not make
  their search exhaustive or computationally equivalent.
- ID-only GRU4Rec/LightGCN and co-occurrence ItemKNN have no metadata-based
  cold-start mechanism. Scores for interaction-unseen IDs are not evidence of
  learned cold-start preference. Inspect warm and popularity-group results too.
- Already-seen targets remain in the denominator, as in existing GenRec
  evaluation. Their count and the count of lists shorter than 50 are reported.
- Exact-score ties are resolved by ascending item ID. GenRec/TIGER retain the
  existing beam implementation's tie behavior.

The neural baselines default to selection by **validation all-target NDCG@10**.
`--selection-metric recall@20` is also available. Existing GenRec checkpoints
were selected by token loss: the report calls out this difference. Hyperparameter
settings below are starting points, not tuned best results. Use validation for
selection, preserve run names, then report the chosen models on test.

## Fit and train

Use a new `--run-name` for each configuration. Existing run directories are
rejected rather than overwritten; omitting the name creates a timestamped run.

```bash
python benchmarks/main-comparison/train.py --model itemknn --run-name itemknn \
  --neighbors 100 --similarity-block-size 128

python benchmarks/main-comparison/train.py --model gru4rec --run-name gru4rec \
  --d-model 128 --n-layers 1 --batch-size 256 --lr 0.001

python benchmarks/main-comparison/train.py --model lightgcn --run-name lightgcn \
  --d-model 64 --n-layers 3 --batch-size 2048 --lr 0.001 \
  --weight-decay 0 --reg-weight 0.0001

python benchmarks/main-comparison/train.py --model tiger --run-name tiger_content \
  --sid-source content --batch-size 128 --lr 0.0003 \
  --eval-batch-size 16 --decode-batch-size 128 --beam-size 100
```

ItemKNN fitting and LightGCN graph propagation can be expensive on the main
catalog. TIGER sees up to 201 encoder tokens for 50 items plus a user token.
Lower batch sizes if needed. `--amp` optionally enables CUDA float16 for GRU4Rec
and TIGER. `--num-workers` defaults to 0; training entry points are guarded for
spawn/forkserver workers.

All runs save `best.pt` inside `benchmarks/main-comparison/checkpoints/<run-name>/`
and `history.json` inside `artifacts/benchmarks/main-comparison/training/<run-name>/`.
Use `--history-dir` to override the history root separately from `--checkpoint-dir`.
Neural histories include
loss, per-group validation metrics, epoch duration, and best epoch. ItemKNN saves
the fitted sparse matrix and validation metrics. Test evaluation is separate.

For a short neural smoke run, add `--smoke --valid-limit 32`. It uses at most
1,024 training examples and one epoch, with a separate `_smoke` directory.
LightGCN still constructs the full training graph in this mode. ItemKNN has no
cheap full-catalog smoke fit; use a small fixture directory with `--data-dir`
instead. To evaluate only a small number of users, use `--limit 10` below.

## What “TIGER” means here

The paper's core design is RQ semantic IDs plus autoregressive sequence-to-sequence
retrieval. Here, input item SIDs are **flattened into a token sequence**, preceded
by a stable hashed-user token (2,000 buckets by default). Each SID position has
its own token range in one shared vocabulary. Token embeddings are learned from
scratch and shared with the output layer; they are not initialized from RQ vectors.

Default T5 dimensions follow the paper's reported encoder-decoder shape: four
layers each, model dimension 128, six heads with dimension 64, feed-forward
dimension 1,024, and dropout 0.1. `--user-buckets 0` removes the user token for
a history-only ablation. These choices differ from GenRec's summed per-item SID
embeddings and per-level output distributions.

This is an **adapted TIGER baseline, not an exact paper reproduction**:

- Reuse this repository's BGE/content RQ-VAE IDs instead of retraining the paper's
  Sentence-T5/32-dimensional tokenizer. Prepare them with the root README pipeline.
- Use Hugging Face T5, the current main Amazon dataset, and this repository's
  all-prefix training/evaluation protocol. The paper used T5X and other datasets.
- Use AdamW, validation-based early stopping, and an epoch budget; the paper's
  original training schedule is not reproduced.
- Generate exactly four SID tokens, without a learned EOS decision, and constrain
  decoding to the catalog trie. All four code ranges reserve at least 256 entries.
- Compute probabilities over the complete shared vocabulary **before** selecting
  a level's token range. The beam-search adapter preserves that normalization.

`--sid-source twotower` is an optional tokenizer-controlled extension; label it
accordingly rather than reporting it as the paper's content tokenizer. SID hashes
are checked when loading checkpoints. Architecture, tokenizer, optimization, and
search choices are recorded; a difference in recall cannot isolate all of them.

## Evaluate the selected checkpoints

```bash
python benchmarks/main-comparison/evaluate.py \
  --checkpoint benchmarks/main-comparison/checkpoints/itemknn/best.pt --split test
python benchmarks/main-comparison/evaluate.py \
  --checkpoint benchmarks/main-comparison/checkpoints/gru4rec/best.pt --split test
python benchmarks/main-comparison/evaluate.py \
  --checkpoint benchmarks/main-comparison/checkpoints/lightgcn/best.pt --split test
python benchmarks/main-comparison/evaluate.py \
  --checkpoint benchmarks/main-comparison/checkpoints/tiger_content/best.pt \
  --split test --beam-size 100 --batch-size 16 --decode-batch-size 128

# Reuse the already-trained GenRec content-C checkpoint, with matching evaluation.
python -m scripts.evaluate_genrec --sid-source content --split test \
  --beam-size 100 --batch-size 64 \
  --output artifacts/benchmarks/main-comparison/evaluation/genrec_content_C_test.json
```

Use `--split valid` for validation comparisons, and `--output` to preserve
alternative evaluation settings. Training consumes no test metrics. The standard
GenRec command uses its default full-vocabulary per-level scores with no prior
correction or trie renormalization.

## Produce a comparison table

```bash
python benchmarks/main-comparison/report.py \
  artifacts/benchmarks/main-comparison/evaluation/itemknn_test.json \
  artifacts/benchmarks/main-comparison/evaluation/gru4rec_test.json \
  artifacts/benchmarks/main-comparison/evaluation/lightgcn_test.json \
  artifacts/benchmarks/main-comparison/evaluation/tiger_content_test_beam100.json \
  artifacts/benchmarks/main-comparison/evaluation/genrec_content_C_test.json
```

This writes `comparison.md` and `.csv` under `artifacts/benchmarks/main-comparison/reports/`.
Markdown uses percentages; CSV retains metric fractions. Mismatched dataset
hashes, split/catalog, ordered evaluation user IDs, target/group counts, and
generative beam widths are rejected.
Smoke/subset outputs require `--allow-smoke` and are labeled accordingly. Old
GenRec JSON without dataset/user signatures must be regenerated with the current
evaluator. These checks do not make unequal tuning budgets a controlled ablation.

Suggested first validation sweeps: ItemKNN neighbors 50/100/200; GRU dimensions
64/128 and learning rates 0.0003/0.001; LightGCN layers 1/2/3 and regularization
0.00001/0.0001/0.001; TIGER learning rate 0.0001/0.0003 and user token on/off.
Change one setting at a time, use separate run names, and record the selection
criterion. No benchmark-quality results have been produced by adding this code.
