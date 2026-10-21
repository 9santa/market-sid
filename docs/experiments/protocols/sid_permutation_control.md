# Semantic assignment control: original versus permuted Two-Tower SIDs

Question: does the learned item-to-SID assignment help GenRec, beyond representing
items with four tokens and the existing catalog trie? This is a new matched pair
of GenRec C runs, not an inference-time substitution in the old checkpoint.

Completed main-validation evidence: [results and interpretation](../results/sid_permutation_results_20261018.md).

## What changes

`scripts.prepare_sid_permutation` copies the existing Two-Tower SIDs to
`item_semantic_ids_twotower_original.npy` and permutes **whole rows** into
`item_semantic_ids_twotower_permuted.npy`. It never retrains the tokenizer.
Both sources use the unchanged `checkpoints/rqvae_twotower_best.pt` codebooks.
The old `twotower` SID file and GenRec checkpoint remain untouched.

Permutation bins use only train interaction counts: exact counts 0 through 20,
then 21–40, 41–80, etc. Unseen items stay in the count-zero bin. This preserves
head/mid/tail/unseen membership while mixing semantic assignments within bins.
Ordinary random permutations can leave fixed points; singleton bins cannot move.
The export reports moved items and unchanged prefixes rather than claiming every
item or prefix changed. The permutation seed is independent of training RNG.

The complete SID set, uniqueness, vocabulary, sequence length, trie topology,
and bucket sizes are unchanged. PAD remains row zero in model lookup tables and
does not participate. Histories and targets both use the new mapping; item IDs,
interactions, evaluation targets, and seen-item filtering remain unchanged.

Train prefix frequencies need not stay identical: head bins contain different
counts, and next-item target counts exclude first items in training histories.
The JSON exports report prefix-frequency changes at all four depths for both
train interactions and actual training targets, including total variation
(`sum(abs(new_count - old_count)) / (2 * total_count)`). Inspect these before
attributing every difference to semantic geometry.

## Prepare and train

From the repository root:

```bash
python -m scripts.prepare_sid_permutation --seed 42

python -m scripts.train_genrec --variant C --sid-source twotower_original \
  --seed 42 --epochs 50 --no-early-stopping

python -m scripts.train_genrec --variant C --sid-source twotower_permuted \
  --seed 42 --epochs 50 --no-early-stopping
```

Exports include `.json` manifests and
`data/processed/item_semantic_ids_twotower_permutation.npy`:
`permuted[row] = original[permutation[row]]`, zero-based, excluding PAD.
Preparation refuses existing control outputs unless `--overwrite` is supplied.
Archive previous control exports/checkpoints/results before changing the
permutation seed: names are fixed, and retraining the same source replaces its
best checkpoint. Training rejects changed SID/data/RQ/permutation hashes.

Both models start from the same seed, architecture, trainable weights, codebook
initialization and tying. `initialization_sha256` excludes the nonpersistent SID
lookup and should match exactly. The same seed and unchanged dataset preserve
the example-shuffle RNG setup; CUDA is not forced into deterministic mode, so
this is not a guarantee of bitwise training reproducibility.

The epoch budget is fixed at 50 for both runs; disabling early stopping does not
change selection: each saves the checkpoint with lowest teacher-forced validation
loss. All other settings come from the same `train_genrec.CONFIG`. The old
checkpoint is historical context, not the matched original baseline. No extra
hyperparameter search is part of this control.

## Evaluate the matched pair

Start with validation and identical decoding settings:

```bash
python -m scripts.evaluate_genrec --sid-source twotower_original \
  --split valid --beam-size 100 --batch-size 64 --diagnose-prefixes

python -m scripts.evaluate_genrec --sid-source twotower_permuted \
  --split valid --beam-size 100 --batch-size 64 --diagnose-prefixes

python -m scripts.report_sid_ablation --permutation-control --split valid
```

The combined report is `artifacts/main/reports/sid_permutation_valid.md`. It checks matching
initial weights, RQ checkpoint, data, train configuration/completed epoch budget,
permutation provenance, catalog prefix counts and ordered evaluation users. It
includes survival at every SID position, Recall@10/50 and NDCG@50 by group.
Use `--limit 10` on both evaluations and the report for a limited smoke run.
Keep full-validation and limited results separate.

Once the protocol is fixed, replace `valid` with `test` in the three commands.
The existing test set has already informed earlier experiments; this does not
turn it into a previously untouched confirmatory holdout.

The sources also work with `evaluate_sid_prefix_retrieval` if per-user ranks and
coarse candidate coverage are needed. Do not mix SID files and checkpoints across
the two conditions.

## Interpretation and stopping rule

- Original beats permuted on rare-item recall: evidence that learned assignment
  helps those groups in GenRec C, even if Content Two-Tower still performs better.
- Original improves only prefix survival/coverage: useful routing structure,
  without demonstrated improvement in final rare-item ranking.
- Little difference: no clear benefit detected for this assignment/model/budget;
  not proof that semantic IDs never help.
- Permuted improves: evidence against the current assignment under this protocol;
  inspect frequency shifts and training curves before choosing a mechanism.

This control measures assignment utility including its alignment with RQ
initialization. It does not isolate hierarchy from initialization, or pure BGE
semantics from behavior learned by the Two-Tower tokenizer. Keeping codebook
vectors fixed is intentional; moving vectors with products would change the
intervention. One permutation and training seed are exploratory evidence. Repeat
only if the effect warrants the budget; do not keep changing seeds until the
preferred conclusion appears.

Verified on a temporary synthetic catalog: export invariants and reproducibility,
two tiny two-epoch training runs with matching initial-state hashes, evaluation,
combined report generation, overwrite rejection, and mismatched RQ-hash rejection.
All 18 existing tests passed; no new test files or dependencies were added. The
user subsequently completed main-data export, training and validation; see the
linked results for measured quality and remaining limitations.
