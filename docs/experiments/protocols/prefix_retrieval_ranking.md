# Prefix retrieval: history controls and learned ranking

Implements step 2 of the [completion plan](../reviews/project_review_20261017.md).
The experiment freezes GenRec, its SID catalog, and the existing Content
Two-Tower. It trains nothing and uses validation for comparison.

## Questions

1. Does GenRec cover targets better than frequency/size-matched random prefixes
   across several random seeds?
2. Does using the correct user's history improve prefix coverage?
3. Can the trained Two-Tower rank covered targets better than a mean item-vector
   history profile?
4. Does prefix restriction help ranking relative to the same Two-Tower scoring
   the full catalog, and what is lost through restricted candidate coverage?
5. Do quota hybrids improve rare-item hits at an acceptable aggregate cost?
6. Do prefix hybrids improve over the same quota filled from full-catalog TT?

## Run

From the repository root, using the existing environment and checkpoints:

```bash
python -m scripts.evaluate_sid_prefix_retrieval \
  --sid-source twotower --split valid --depths 2 3 \
  --limit 10000 --seed 42 \
  --beam-size 100 --batch-size 16 --decode-batch-size 512 \
  --control-seeds 42 43 44 \
  --shuffle-histories --shuffle-seed 42 \
  --two-tower-checkpoint checkpoints/two_tower_best.pt \
  --hybrid-slot-sweep 5 10 \
  --output artifacts/main/diagnostics/prefix_retrieval/prefix_retrieval_valid10000_learned_tt.json
```

For a first small execution, use `--limit 16` and a different output filename.
Use `--limit 0` for all validation users. Keep the sample seed and limit fixed
when comparing configurations. The 10k command uses the same sampling procedure
as the earlier v2 10k reports, provided the dataset is unchanged.

One run evaluates both quotas and all matched-control seeds. GenRec runs once
per real-history batch and once per shuffled-history batch, not once per quota
or matched seed. Full-catalog Two-Tower scores are computed once per batch and
reused for each depth. The trained tower uses its checkpoint's history length;
the mean-profile baseline keeps `--history-window` (default 20).

The learned tower always reads validated `item_text_embeddings.npy` and computes
its own item vectors from the selected checkpoint. `--content-embeddings` changes
only the separate mean-profile baseline. Optionally point it to
`data/processed/item_two_tower_embeddings.npy` and save another output to compare
mean TT vectors against the trained TT user encoder.

A cheaper controls-only run skips content ranking and Two-Tower:

```bash
python -m scripts.evaluate_sid_prefix_retrieval \
  --sid-source twotower --split valid --depths 2 3 \
  --limit 10000 --seed 42 --beam-size 100 \
  --control-seeds 42 43 44 --shuffle-histories --shuffle-seed 43 \
  --coverage-only \
  --output artifacts/main/diagnostics/prefix_retrieval/prefix_retrieval_valid10000_controls_shuffle43.json
```

This also checks sensitivity to a second history permutation without retraining.
It still performs beam search. Do not combine `--coverage-only` with
`--two-tower-checkpoint`.

## Rankings in the JSON

| Method | Candidate source and score |
|---|---|
| `exact` | Ordinary full-SID GenRec beam, filtered for seen items; not exhaustive item scoring |
| `two_tower_full` | All catalog items ranked by trained `encode(history) @ item_vectors.T` |
| `hybrid_full_two_tower_slots5/slots10` | Exact top-45/top-40 plus novel full-catalog TT items, without prefix restriction |
| `prefix2_only`, `prefix3_only` | Expanded prefixes ranked by their generation score |
| `prefix{d}_content` | Prefix score plus cosine with the mean history item vector |
| `prefix{d}_normalized` | Prefix score minus log of full bucket size before seen filtering |
| `prefix{d}_normalized_content` | Normalized prefix score plus mean-profile cosine |
| `prefix{d}_two_tower` | Same expanded candidates, ranked solely by the learned Two-Tower score |
| `hybrid{d}_{variant}_slots5/slots10` | Exact top-45/top-40 plus up to 5/10 novel items from the expanded ranking |

Hybrids use normalized, normalized-content, and learned-Two-Tower variants.
Novel items must be outside the complete exact top-50. If too few are available,
the remaining exact results fill the list. Therefore top-10/20 are unchanged for
quotas 5/10 by construction; compare Recall/NDCG@50 and absolute hits by group.
Stable item-ID tie-breaking is used for all reranked lists. PAD and the complete
recipient history are excluded everywhere; already-seen targets remain in the
denominator and are counted explicitly.

Each `prefix{d}_*` ranking reports `candidate_hits`, `candidate_coverage`, and
`recall@{k}_given_covered`. The last is hits@K divided by candidate hits, or null
if no target was covered. For example, high coverage with low conditional recall
isolates a ranking failure. Do not compare conditional recall alone across depths
without also reporting the coverage and size of their different candidate pools.

## Full-TT hybrid control and paired results (schema v4)

With `--two-tower-checkpoint`, full-TT hybrids run automatically for the same
`--hybrid-slot-sweep` quotas as prefix hybrids. Both exclude the entire exact
top-50 from additions and use the same seen-item filtering, ordering and backfill.
Full-TT retrieval keeps its first 100 candidates internally so overlap with exact
cannot exhaust the requested novel additions. The standalone TT metric still
uses only its first 50. No extra model forward pass is needed.

To compare on the full validation set without repeating the shuffled-history
experiment:

```bash
python -m scripts.evaluate_sid_prefix_retrieval \
  --sid-source twotower --split valid --depths 2 3 \
  --limit 0 --seed 42 \
  --beam-size 100 --batch-size 256 --decode-batch-size 512 \
  --two-tower-checkpoint checkpoints/two_tower_best.pt \
  --hybrid-slot-sweep 5 10 \
  --output artifacts/main/diagnostics/prefix_retrieval/prefix_retrieval_valid_full_tt_control.json
```

The JSON includes `paired_comparisons` for each full-TT hybrid versus exact,
and each prefix-TT hybrid versus the full-TT hybrid with the same quota. All
deltas are **method minus reference**, in fractions rather than percentage
points. Reports include Recall/NDCG@10/20/50 by group, method-only/reference-only
hits, standard errors, and approximate pointwise 95% normal intervals calculated
from per-user paired differences (`mean ± 1.96 * sample_std / sqrt(n)`).
These intervals are not corrected for multiple comparisons or training variation.
They are unreliable for sparse differences; an all-zero observed difference
gives `[0, 0]`, which does not establish population equivalence. Inspect the
discordant hit counts and `nonzero_differences`, especially for unseen items.

Every run also writes `<output_stem>_users.npz`; the JSON records its path and
SHA-256. It contains `user_idx`, `target`, `group`, `methods`, and a
`target_rank` matrix (rows = users, columns = methods). Ranks are 1–50, with
0 meaning absent from top-50, including seen targets. These are sufficient to
recompute Recall/NDCG or bootstrap paired users without repeating inference:

```python
import numpy as np

rows = np.load("artifacts/main/diagnostics/prefix_retrieval/prefix_retrieval_valid_full_tt_control_users.npz")
column = rows["methods"].tolist().index("hybrid_full_two_tower_slots5")
ranks = rows["target_rank"][:, column]
hits50 = ranks > 0
```

Top-10/20 remain identical to exact for quotas 5/10; the informative comparison
is at 50. Select configurations on validation. A small positive aggregate delta
alone does not establish a prefix-specific gain, particularly after trying
multiple depths and quotas.

## Controls

- `popularity_controls.top_train`: globally most frequent training-target
  prefixes, with the same prefix count as the real beam.
- `popularity_controls.matched_train_size_seed42/43/44`: random selection within
  joint train-frequency/catalog-size bins. No test/validation target frequency
  enters the control. Original prefixes remain eligible.
- `matched_control_summary`: mean coverage and standard deviation across seeds.
  This deviation describes random-control variation, not a user-level confidence
  interval. Per-seed reports retain paired wins/losses and overlap.
- `shuffled_history_control`: replace each selected user's GenRec input with
  another selected user's history. A seeded cycle has no self-pairs and reuses
  every donor once. Targets and seen filtering remain attached to the recipient.
  Two-Tower ranking always uses the real recipient history. The control pools
  history lengths and categories; it measures dependence on the supplied history,
  not the isolated causal effect of semantic IDs.

Both control types report `learned_only_hits`, `control_only_hits`, coverage
differences, pool sizes, and prefix overlap. Inspect these together: fixed beam
width does not imply a fixed number of expanded item candidates.

## Provenance, costs, and interpretation

Schema v4 retains checkpoint/data/SID/embedding hashes, ordered-user and shuffle
donor hashes, seeds, quotas, batch settings, and evaluator source hash. The
Two-Tower loader checks the catalog mapping and the embedding cache. Legacy
Two-Tower checkpoints do not fingerprint their training dataset/features;
`training_data_signature_verified=false` makes that limitation explicit.

`timing_seconds` separates real/shuffled GenRec, Two-Tower encoding/full scoring,
and CPU ranking/control work. `cuda_peak_allocated_bytes` includes resident
models/features during evaluation, not total GPU reservation or CPU memory.
These are diagnostic costs. The prefix ranker reuses full-catalog scores for
consistent quality comparisons; this implementation does not demonstrate a
standalone prefix-reranking latency advantage.

Interpret results in order: real versus control coverage; learned versus
mean-profile conditional recall on identical candidates; prefix TT versus full
TT; finally hybrids versus exact GenRec. If TT alone explains the improvement,
do not credit semantic retrieval. Keep negative results and head/rare trade-offs.

Implementation was checked with temporary synthetic CPU fixtures, including a
seen target, direct full-TT scoring, quota preservation, batching invariance,
legacy method equivalence, and checkpoint mapping rejection. The 18 existing
tests passed. No new test files or dependencies were added. Main-dataset v3
quality results were subsequently supplied by the user and are analyzed in the
[full-validation review](../results/prefix_retrieval_full_valid_20261017.md).

The v4 full-TT hybrid control and per-user output were checked on small synthetic
models, including direct full-catalog scoring, overlap/backfill, seen filtering,
reconstruction of aggregate metrics from saved ranks, and paired-difference
calculations. The 18 existing tests passed. The user's completed full-validation
v4 run is analyzed in the [full-TT control review](../results/full_tt_control_20261018.md),
including exact paired checks for sparse unseen hits.
