# Final benchmark: all GenRec variants

Before final test, the separate [target-weighted training experiment](target_weighted_genrec.md)
compares gamma 0/0.25/0.5 on validation. Its checkpoints are kept outside this matrix.

This adds A/B/D to the [original final protocol](final_benchmark.md), keeping C
and every existing data, tokenizer, training, selection, and retrieval setting.
Use [final_protocol_all_variants.json](../../../benchmarks/main-comparison/final_protocol_all_variants.json)
for the expanded comparison. It records the original protocol's SHA-256; changing
the original file invalidates compatibility rather than silently adopting new rules.

| Variant | Semantic input embeddings | Semantic output weights | Additional loss |
|---|---|---|---|
| A | Random initialization | Independent | None |
| B | RQ codebook initialization, trainable | Independent | None |
| C | RQ codebook initialization, trainable | Tied to input embeddings | None |
| D | Fixed RQ base + trainable delta | Tied to effective input embeddings | `0.01 × anchor_loss` |

D uses the existing model's relative squared displacement penalty to keep the
learned embeddings near their RQ bases. A still consumes the same semantic IDs;
only its embedding initialization differs. Its RQ checkpoint is read to validate
the tokenizer and full vocabulary sizes, but its vectors are not installed.
These are the same A/B/C/D definitions as `scripts/train_genrec.py`.

Each variant uses both `content` and `twotower` SIDs and training seeds 42, 43, 44:
**24 GenRec runs**, plus 16 baseline runs = **40 checkpoints**. SIDs and the
seed-42 teacher/tokenizers stay fixed across these runs. All models select by
full-validation **Recall@50**, patience 5, maximum 50 epochs. D's anchor penalty
changes the training objective, not the checkpoint-selection criterion.

Existing completed C/baseline runs under the original protocol are accepted.
Missing `variant` in an old GenRec checkpoint means C. No dataset, tokenizer,
checkpoint, or training history needs to be rewritten. An unfinished run must
finish normally before selection can be frozen; this change does not add resume.

The primary comparison remains **GenRec C, Two-Tower SIDs vs Content Two-Tower**.
A/B/D are additional ablations. Report all variants and seeds; do not replace C
with whichever variant wins on test. If test results have already been inspected,
document that fact and label the added comparisons exploratory.

## Add A/B/D to existing runs

Run from the repository root in the activated environment. These commands train
on CUDA. They do not repeat data preparation or completed C/baseline training.
Run directories must be new; stop on failure and inspect the error.

```bash
BENCH=benchmarks/main-comparison
PROTOCOL=$BENCH/final_protocol_all_variants.json
FINAL_DATA=data/final-v1
FINAL_CKPT=$BENCH/checkpoints/final-v1
FINAL_ART=artifacts/benchmarks/final-main

for variant in A B D; do
  for source in content twotower; do
    for seed in 42 43 44; do
      python "$BENCH/train.py" --protocol "$PROTOCOL" --data-dir "$FINAL_DATA" \
        --checkpoint-dir "$FINAL_CKPT" --history-dir "$FINAL_ART/training" \
        --model genrec --variant "$variant" --sid-source "$source" \
        --run-name "genrec_${source}_${variant}_s${seed}" --seed "$seed" \
        --rq-checkpoint "$FINAL_CKPT/tokenizers/rqvae_${source}_best.pt" \
        --device cuda --eval-batch-size 16 --decode-batch-size 128 || break 3
    done
  done
done
```

The protocol fixes D's `--anchor-lambda` to 0.01 automatically; a conflicting
explicit value is rejected. Histories/checkpoints record variant and anchor weight.

For a fresh benchmark, prepare data/features and baselines with the original
guide first. To train all GenRec variants from scratch, change the loop to
`for variant in A B C D`. Do not rerun C if you already have it: old names such as
`genrec_content_s42` and new names such as `genrec_content_C_s42` both represent the
same condition. Freezing rejects duplicate condition/seed pairs.

If only some C runs are missing, launch them individually, for example:

```bash
python "$BENCH/train.py" --protocol "$PROTOCOL" --data-dir "$FINAL_DATA" \
  --checkpoint-dir "$FINAL_CKPT" --history-dir "$FINAL_ART/training" \
  --model genrec --variant C --sid-source content --run-name genrec_content_s44 \
  --seed 44 --rq-checkpoint "$FINAL_CKPT/tokenizers/rqvae_content_best.pt" \
  --device cuda --eval-batch-size 16 --decode-batch-size 128
```

Complete all six C runs and the baseline runs (including three TIGER seeds) using
the original guide. Both protocols accept C; A/B/D require the expanded protocol.

## Freeze, evaluate, report

If some planned seeds could not be trained, add `--allow-missing-seeds` to
`freeze_selection.py` and pass only completed, protocol-compatible checkpoints.
The selection records every missing condition/seed before test. It still rejects
duplicates, incomplete runs, changed training settings, and weighted-loss ablations.
Report all available seeds; do not select runs by their test results. The report
lists actual seeds and uses SD N/A for single runs. Paired comparisons use only
shared seeds, so they may use fewer runs than the aggregate table.

Only after all 40 runs finish, freeze the validation-selected checkpoints. The
command checks completeness, duplicates, protocol/data identity, and completed
training histories. The tokenizer directory contains no `best.pt` and is excluded
by this glob. Keep unrelated runs outside this final checkpoint directory.

```bash
python "$BENCH/freeze_selection.py" "$FINAL_CKPT"/*/best.pt \
  --protocol "$PROTOCOL" --data-dir "$FINAL_DATA" \
  --output "$FINAL_ART/selection_all_variants.json"
```

Use a separate selection and evaluation directory to preserve any earlier results.
Evaluations record both the training protocol and the expanded evaluation protocol.

```bash
for checkpoint in "$FINAL_CKPT"/*/best.pt; do
  run_name=$(basename "$(dirname "$checkpoint")")
  python "$BENCH/evaluate.py" --checkpoint "$checkpoint" --protocol "$PROTOCOL" \
    --data-dir "$FINAL_DATA" --selection-file "$FINAL_ART/selection_all_variants.json" \
    --split test --device cuda --beam-size 100 --batch-size 16 --decode-batch-size 128 \
    --output "$FINAL_ART/evaluation_all_variants/${run_name}_test.json" || break
done

python "$BENCH/final_report.py" "$FINAL_ART"/evaluation_all_variants/*_test.json \
  --selection-file "$FINAL_ART/selection_all_variants.json" \
  --output "$FINAL_ART/reports/all_variants.md"
```

The report separates source/variant conditions, aggregates all three training
seeds, and retains the existing paired per-user comparisons against Two-Tower.
It requires exactly one evaluation per selected checkpoint.
