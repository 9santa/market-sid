# Main dataset: SID depth and Two-Tower supervision

Commands for the SID-depth comparison. Background is in the [research log](../research_log.md#6-four-learned-codes-and-uniqueness-aware-export--h07); measured outcomes are in [results](../results/overview.md#main-sid-comparison).

All runs share the main dataset, frozen Two-Tower item vectors, and GenRec **variant C**. The SID versions A/B/C below are tokenizer labels, not GenRec variants.

| SID version | RQ input | Extra tokenizer supervision | SID structure | `--sid-source` |
|---|---|---|---|---|
| Current | Two-Tower vectors | None; input already behavior-adapted | 3 RQ + collision | `twotower` |
| A | Same Two-Tower vectors | None | 4 RQ | `rq4_a` |
| B | BGE vectors | Predict frozen Two-Tower vector from final RQ sum | 4 RQ | `rq4_b` |
| C | Same BGE vectors | Same prediction at every partial RQ sum | 4 RQ | `rq4_c` |

For B/C, a linear projection maps quantized sums to the Two-Tower teacher dimension. Loss: reconstruction MSE + VQ loss + `behavior_weight * mean(1 - cosine(prediction, teacher))`. B supervises depth 4; C averages depths 1–4. They use the same weight and parameter count. The auxiliary loss trains the encoder and projection; codebooks are updated by VQ loss.

Tokenizer defaults: 30 epochs, batch 1,024, Adam 0.001, seed 42, hidden/latent widths 256/128, four 256-entry codebooks, beta 0.25, behavior weight 0.05. The behavior weight is not tuned. B/C use the same value, and tokenizers are selected by reconstruction loss. GenRec training and checkpoint selection are unchanged.

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

Export gives items in each `[c1,c2,c3]` bucket distinct fourth codes. It keeps the first three codes fixed and minimizes squared residual-to-codeword distance through bipartite matching. An already-unique nearest-code assignment needs no change.

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

For Current, “full SID” means three RQ tokens plus a collision suffix; for A/B/C it means four RQ tokens. Early prefix loss locates pruning but does not, alone, identify its cause.
