# Main dataset: SID depth and Two-Tower supervision

This page is the runnable protocol. See the [research log](../research_log.md#6-four-learned-codes-and-uniqueness-aware-export--h07)
for the original hypothesis and subsequent interpretation, and the
[saved results](../results/overview.md#main-sid-comparison) for the completed comparisons.

All runs use the existing main dataset in `data/processed`, frozen exported
Two-Tower vectors, and GenRec architecture **C**. SID variant letters below are
separate from the GenRec `--variant` argument, which always stays `C`.

| SID version | RQ input | Extra tokenizer supervision | SID structure | `--sid-source` |
|---|---|---|---|---|
| Current | Two-Tower vectors | None; input already behavior-adapted | 3 RQ + collision | `twotower` |
| A | Same Two-Tower vectors | None | 4 RQ | `rq4_a` |
| B | BGE vectors | Predict frozen Two-Tower vector from final RQ sum | 4 RQ | `rq4_b` |
| C | Same BGE vectors | Same prediction at every partial RQ sum | 4 RQ | `rq4_c` |

For B/C, a shared linear projection maps each supervised quantized sum to the
teacher dimension. The loss is reconstruction MSE + existing VQ loss +
`behavior_weight * mean(1 - cosine(prediction, teacher))`. B uses depth 4;
C averages depths 1–4, keeping the overall weight and parameter count identical.
Both use the existing straight-through estimator: the auxiliary loss trains the
encoder and projection; codebook vectors are updated through the VQ loss.
This encourages prefix-level predictability, but does not guarantee locality.

Defaults: 30 tokenizer epochs, batch 1024, Adam lr 0.001, seed 42,
hidden/latent dimensions 256/128, four 256-entry codebooks, beta 0.25,
behavior weight 0.05. This weight is an experimental starting point, not a tuned
value. Keep it identical for B and C. Select each tokenizer by reconstruction
loss, matching the current tokenizer's rule. GenRec retains its existing
Transformer, optimizer, split, and validation selection settings. All four
RQ levels receive initialization and input/output tying in A/B/C.

## Run from the repository root

Reuse the existing `item_two_tower_embeddings.npy`, `rqvae_twotower_best.pt`,
and `item_semantic_ids_twotower.npy` for Current. If these are not prepared:

```bash
python -m scripts.export_two_tower_embeddings
python -m scripts.train_twotower_rqvae
python -m scripts.export_twotower_semantic_ids
```

Train each new tokenizer and export its IDs. These commands do not modify the
current tokenizer. Each saves a checkpoint, training history and export report.

```bash
python -m scripts.train_sid_ablation --variant A
python -m scripts.train_sid_ablation --variant B --behavior-weight 0.05
python -m scripts.train_sid_ablation --variant C --behavior-weight 0.05
```

Export enforces unique fourth-level codes within each `[c1,c2,c3]` bucket.
It computes each item's residual after subtracting its first three codewords,
then solves a minimum-cost bipartite assignment using squared distances to the
fourth codebook. The first three codes remain fixed. Matching covers every item
in a bucket with collisions; already-unique nearest-code assignments are optimal
and can be kept. This same export policy applies to A, B and C.

A bucket larger than 256 items cannot receive distinct codes from a 256-entry
fourth codebook. Export stops with the offending prefix and size in this case.
Inspect `artifacts/main/tokenizers/rqvae_rq4_*_export.json` for raw/final collision rates, largest
bucket, changed assignments and added squared quantization error. Assignment is
performed during export; tokenizer training still uses ordinary RQ quantization.
An added item may require rematching its bucket, so the resulting mapping is
catalog-dependent. Regenerate GenRec checkpoints when the exported IDs change.

Existing tokenizer checkpoints can be re-exported without tokenizer training:

```bash
python -m scripts.train_sid_ablation --variant A --export-only
python -m scripts.train_sid_ablation --variant B --export-only
python -m scripts.train_sid_ablation --variant C --export-only
```

Train the four recommenders with identical settings (skip the first command if
the matching Current GenRec C checkpoint already exists):

```bash
python -m scripts.train_genrec --variant C --sid-source twotower
python -m scripts.train_genrec --variant C --sid-source rq4_a
python -m scripts.train_genrec --variant C --sid-source rq4_b
python -m scripts.train_genrec --variant C --sid-source rq4_c
```

Use the same beam size and all test users for each survival table:

```bash
python -m scripts.evaluate_genrec --sid-source twotower --split test --beam-size 100 --batch-size 64 --diagnose-prefixes
python -m scripts.evaluate_genrec --sid-source rq4_a --split test --beam-size 100 --batch-size 64 --diagnose-prefixes
python -m scripts.evaluate_genrec --sid-source rq4_b --split test --beam-size 100 --batch-size 64 --diagnose-prefixes
python -m scripts.evaluate_genrec --sid-source rq4_c --split test --beam-size 100 --batch-size 64 --diagnose-prefixes
python -m scripts.report_sid_ablation
```

For a smoke run, add `--limit 10` to all four evaluation commands and the report
command. Limited results have separate filenames.

The combined report is `artifacts/main/reports/sid_ablation_test.md`. It checks matching data
signatures, GenRec settings, beam sizes and group counts. Each SID version gets
the same table: group, target count, survival at positions 1/2/3/4, and item top-10.
Groups are all/head/mid/tail/unseen; popularity comes only from training counts.
Survival uses all targets in the group and is measured after each beam pruning,
before seen-item filtering. Top-10 is measured after filtering. Catalog prefix
counts make it explicit when beam 100 retains every first branch.

The full-SID column means three semantic tokens plus collision for Current,
and four semantic tokens for A/B/C. Early target loss locates pruning; it does
not, by itself, establish which training or tokenizer choice caused it.
