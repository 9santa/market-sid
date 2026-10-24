# GenRec target-frequency weighted training

Status: implemented, not run. This is a separate validation experiment before
opening final test. The original 40-condition benchmark and its protocol files
remain unchanged. Its primary comparison still uses unweighted GenRec C.

## Question and fixed setup

Does reducing frequent targets' contribution to the training objective improve
mid/tail retrieval, with the same semantic IDs and decoding?

Use **GenRec C + Two-Tower SIDs**, seeds **42/43/44**, beta **1**, and gamma
**0, 0.25, 0.5**. Reuse completed C runs for gamma=0; train the other six runs from
scratch, not from a selected C checkpoint. Initialization, minibatch shuffling,
architecture, optimizer, SID exports, teacher, train/validation split, and decoding
settings remain matched for each seed. There is no resampling or inference correction.

[target_weighting_experiment.json](../../../benchmarks/main-comparison/target_weighting_experiment.json)
records these conditions and the hash of the expanded final protocol. Both files
are checked by training/evaluation; experiment identity is saved in the outputs.

Keep the existing **validation all-target Recall@50** selection, patience 5,
maximum 50 epochs, and beam 100. This matches the budget cap and stopping rule;
different objectives may select different epochs and stop at different times.
Selection and evaluation metrics are **unweighted**. Changing selection to tail
Recall would confound this objective ablation.

## Exact loss

For item i, n_i counts its occurrences as a positive next-item target among the
actual all-prefix training examples. Each user's first training item contributes
history but is never a positive target. Validation/test events contribute nothing.
These counts differ from the interaction counts used for head/mid/tail grouping;
the existing groups stay unchanged.

Let N = sum_i n_i and L be the SID length. Use:

```text
raw_weight_i = (n_i + beta)^(-gamma)
Z = sum_i n_i * raw_weight_i / N
weight_i = raw_weight_i / Z

loss = mean_examples(weight_target * mean_SID_positions(token_cross_entropy))
```

Z is fixed from the full training set, not recomputed per minibatch. This keeps
the expected weight of a sampled training example equal to one, preserving the
overall loss scale while changing relative emphasis. The same example weight
applies to all four SID positions, including the collision token. This is not
per-token-class weighting. Anchor regularization, if used outside this C experiment,
is kept separate from the weighted CE.

Gamma=0 takes the original CE code path exactly, including its reduction order.
For positive gamma, weights are reconstructed from the fingerprinted train split
when loading a model. They are not trainable parameters and do not change inference.
Logs record target counts and min/max normalized weights.

Items with no positive training targets receive no direct weighted gradient.
In particular, weighting introduces no new unseen-item training examples. Any
unseen benefit would still require generalization through shared representations.

## Train the six additional runs

Activate the existing environment and run from the repository root:

```bash
BENCH=benchmarks/main-comparison
PROTOCOL=$BENCH/final_protocol_all_variants.json
EXPERIMENT=$BENCH/target_weighting_experiment.json
FINAL_DATA=data/final-v1
FINAL_CKPT=$BENCH/checkpoints/final-v1
WEIGHT_CKPT=$BENCH/checkpoints/target-weighting-v1
WEIGHT_ART=artifacts/benchmarks/target-weighting-v1

for gamma in 0.25 0.5; do
  for seed in 42 43 44; do
    python "$BENCH/train.py" --protocol "$PROTOCOL" --loss-experiment "$EXPERIMENT" \
      --data-dir "$FINAL_DATA" --checkpoint-dir "$WEIGHT_CKPT" \
      --history-dir "$WEIGHT_ART/training" \
      --model genrec --variant C --sid-source twotower \
      --target-gamma "$gamma" --target-beta 1 --seed "$seed" \
      --run-name "genrec_twotower_C_g${gamma}_s${seed}" \
      --rq-checkpoint "$FINAL_CKPT/tokenizers/rqvae_twotower_best.pt" \
      --device cuda --eval-batch-size 16 --decode-batch-size 128 || break 2
  done
done
```

The baseline C checkpoints must use the same final dataset and tokenizer. Use all
three preselected seed runs; do not choose among duplicate runs using their scores.
If a baseline seed is missing, train it with the existing unweighted C command.

## Compare on full validation

The baseline paths below use the explicit `genrec_twotower_C_s42` naming scheme.
If your selected controls use the older `genrec_twotower_s42` names, remove `_C`
from the checkpoint path. Specify the corresponding path separately if your seed
runs use mixed naming. No baseline retraining is required.

```bash
for seed in 42 43 44; do
  python "$BENCH/evaluate.py" \
    --checkpoint "$FINAL_CKPT/genrec_twotower_C_s${seed}/best.pt" \
    --protocol "$PROTOCOL" --loss-experiment "$EXPERIMENT" --data-dir "$FINAL_DATA" \
    --split valid --device cuda --beam-size 100 --batch-size 16 --decode-batch-size 128 \
    --output "$WEIGHT_ART/evaluation/genrec_twotower_C_g0_s${seed}_valid.json" || break
done

for gamma in 0.25 0.5; do
  for seed in 42 43 44; do
    python "$BENCH/evaluate.py" \
      --checkpoint "$WEIGHT_CKPT/genrec_twotower_C_g${gamma}_s${seed}/best.pt" \
      --protocol "$PROTOCOL" --loss-experiment "$EXPERIMENT" --data-dir "$FINAL_DATA" \
      --split valid --device cuda --beam-size 100 --batch-size 16 --decode-batch-size 128 \
      --output "$WEIGHT_ART/evaluation/genrec_twotower_C_g${gamma}_s${seed}_valid.json" || break 2
  done
done

python "$BENCH/report.py" "$WEIGHT_ART"/evaluation/*_valid.json \
  --output "$WEIGHT_ART/reports/validation.md"
```

The comparison table lists each gamma/seed separately. Evaluations also save
per-user ranks in NPZ for subsequent paired comparisons. Inspect all nine runs;
the generic comparison report does not enforce the full experiment matrix.

## Interpretation and next decision

Compare Recall/NDCG@10/20/50 for all/head/mid/tail/unseen, averaging over seeds and
checking whether the direction is consistent. Inspect head losses alongside rare
item gains, and compare each weighted seed with the corresponding gamma=0 seed.
Use paired user differences to distinguish small changes from sampling variation.
Training losses across gamma are different objectives and should not be ranked
as if they were comparable measures of recommendation quality.

An improvement would support a contribution from objective weighting to the
observed bias, not prove that popularity priors are the only cause. A null result
under these two gamma values would not prove that better tail learning is impossible.
This experiment alone does not isolate which SID depth changes.

These validation outcomes also guided checkpoint selection, so they remain
development evidence. After reviewing them, document any proposed extension of
the final test matrix before running test. Weighted checkpoints cannot silently
replace C in `freeze_selection.py`, and the experiment evaluator currently accepts
validation only. Existing benchmark checkpoints/results are not overwritten.
