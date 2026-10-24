# Main dataset: ItemKNN, GRU4Rec, LightGCN, and TIGER

Run ItemKNN, GRU4Rec, LightGCN, and TIGER against GenRec on the existing Amazon split. Commands run from the repository root. Nothing is downloaded or trained until a command is run.

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

GRU4Rec and LightGCN use RecBole model classes with this project's data and training loops. LightGCN builds adjacency without the removed SciPy `dok_matrix._update` API, keeping symmetric degree normalization. Negatives exclude a user's training items; validation edges never enter the graph.

The reviewed environment used RecBole 1.2.1, Transformers 5.16.1, PyTorch, NumPy, SciPy, and Polars. Scripts record installed versions but do not install packages. T5 uses numeric tokens, not a pretrained LM or SentencePiece.

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

Neural baselines select by validation **all-target NDCG@10** by default; `--selection-metric recall@20` is available. GenRec instead selected checkpoints by token loss. Treat these as different selection rules, not matched tuning budgets.

## Fit and train

Give each run a new `--run-name`; existing run folders are not overwritten. With no name, the script uses a timestamp.

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

ItemKNN fitting and LightGCN propagation may be slow at this catalog size. TIGER accepts up to 201 encoder tokens for 50 items and a user token. Reduce batch size if memory is tight. `--amp` enables optional CUDA float16 for GRU4Rec and TIGER.

Best weights go to `benchmarks/main-comparison/checkpoints/<run-name>/best.pt`; histories go to `artifacts/benchmarks/main-comparison/training/<run-name>/history.json`. Use `--history-dir` and `--checkpoint-dir` to change either root. Test evaluation is a separate step.

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

This is an **adaptation of TIGER**, not a paper reproduction:

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

The report writes `comparison.md` (percentages) and `.csv` (fractions) in `artifacts/benchmarks/main-comparison/reports/`. It rejects mismatched dataset hashes, splits, catalogs, users, group counts, and beam widths. Use `--allow-smoke` for subset results. Regenerate old GenRec JSON if it lacks dataset/user signatures.

Possible validation sweeps: ItemKNN neighbors 50/100/200; GRU width 64/128 and learning rate 0.0003/0.001; LightGCN 1/2/3 layers and regularization 0.00001/0.0001/0.001; TIGER learning rate 0.0001/0.0003 and user token on/off. Change one setting per run and record the selection metric.
