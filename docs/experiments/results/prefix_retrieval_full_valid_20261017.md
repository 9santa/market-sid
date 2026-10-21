# Full-validation prefix retrieval with learned Two-Tower ranking

Reviewed 2026-10-17. Source: [raw artifact](../../../artifacts/main/diagnostics/prefix_retrieval/prefix_retrieval_valid_nolimit_learned_tt.json).

Artifact SHA-256: `f4fef190a78909088fabb0e07700fa358c87b4dc52db30078a26c0cd8db476f5`.

## Protocol

80,471 validation targets; full catalog; complete recipient-history filtering; beam 100; depths 2/3; matched-control seeds 42/43/44; one shuffled-history seed 42; hybrid quotas 5/10. All reported lists have 50 items and no target is already seen. These are validation results, not the earlier test results.

GenRec: epoch 49, SHA-256 `a20f0c77f77f3fe178d4d12102b6347aa04983da6cf9abcfe454fd51a2030cec`.
Two-Tower: epoch 35, SHA-256 `bf3c6626e10d1c0e62595287ed75c69e318135cebaef96ea34fd590744d17fd6`.

The evaluator source hash matches the current script. Aggregate group counts, hits/recall, conditional recall, and unchanged hybrid top-10/20 were checked. No model inference or training was rerun. The legacy Two-Tower checkpoint has no verified training-data signature; catalog mapping and current embedding cache were checked by the evaluator.

## Main ranking results

Recall@50 in percent; parentheses contain target hits. `exact` is full-SID beam retrieval, not exhaustive scoring.

| Method | All | Head | Mid | Tail | Unseen |
|---|---:|---:|---:|---:|---:|
| exact | 6.619 (5326) | 21.645 (5166) | 0.595 (118) | 0.112 (36) | 0.134 (6) |
| two_tower_full | 5.143 (4139) | 14.794 (3531) | 1.577 (313) | 0.768 (248) | 1.049 (47) |
| prefix2_content | 3.043 (2449) | 8.200 (1957) | 1.275 (253) | 0.654 (211) | 0.625 (28) |
| prefix2_normalized_content | 4.162 (3349) | 12.352 (2948) | 1.124 (223) | 0.483 (156) | 0.491 (22) |
| prefix2_two_tower | 5.258 (4231) | 15.436 (3684) | 1.562 (310) | 0.651 (210) | 0.602 (27) |
| prefix3_two_tower | 5.898 (4746) | 18.130 (4327) | 1.451 (288) | 0.372 (120) | 0.245 (11) |
| hybrid2_two_tower_slots5 | 6.620 (5327) | 21.482 (5127) | 0.680 (135) | 0.180 (58) | 0.156 (7) |
| hybrid2_two_tower_slots10 | 6.539 (5262) | 20.954 (5001) | 0.852 (169) | 0.251 (81) | 0.245 (11) |
| hybrid3_two_tower_slots5 | 6.636 (5340) | 21.574 (5149) | 0.685 (136) | 0.155 (50) | 0.112 (5) |

GenRec leads aggregate and head metrics. Full-catalog Content Two-Tower leads mid/tail/unseen. Prefix-2 TT has 92 more aggregate hits than full TT, entirely explained by head (+153), offset by mid (-3), tail (-38), and unseen (-20). Thus prefix restriction does not improve rare-item retrieval over the same full-catalog scorer in this run.

## Coverage controls

Candidate coverage in percent. Matched is the mean across three random-control seeds; it is not a confidence interval.

| Depth | Group | n | Real history | Top train | Matched | Shuffled history |
|---|---|---:|---:|---:|---:|---:|
| 2 | all | 80471 | 21.809 | 11.984 | 9.338 | 8.232 |
| 2 | head | 23867 | 41.480 | 30.867 | 23.697 | 21.075 |
| 2 | mid | 19843 | 17.039 | 6.259 | 4.396 | 4.208 |
| 2 | tail | 32279 | 11.726 | 2.900 | 2.722 | 2.128 |
| 2 | unseen | 4482 | 10.799 | 2.209 | 2.410 | 1.606 |
| 3 | all | 80471 | 10.651 | 6.435 | 4.511 | 4.031 |
| 3 | head | 23867 | 30.951 | 20.891 | 14.240 | 12.838 |
| 3 | mid | 19843 | 3.845 | 0.670 | 0.729 | 0.595 |
| 3 | tail | 32279 | 1.177 | 0.170 | 0.241 | 0.177 |
| 3 | unseen | 4482 | 0.915 | 0.089 | 0.201 | 0.112 |

At depth 2 the mean real-history pool is 2,938 items, matched pools average 2,917, and shuffled pools 2,939. Top-train selects a larger pool (4,548). The large real-history advantage is not explained by a larger candidate pool relative to matched/shuffled controls.

For tail at depth 2, real history covers 3,785 targets versus 687 with shuffled history; paired real-only/control-only counts are 3,351/253. For unseen these are 484 versus 72, with paired counts 429/17. This supports history-dependent retrieval of regions containing rare items. It does not isolate semantic-ID causality: the shuffle crosses categories/history lengths and there is no random-SID trained baseline.

## Ranking within covered candidates

Recall@50 conditional on candidate coverage, percent. These compare identical candidate pools within a depth.

| Depth | Group | Covered targets | Prefix + mean content | Normalized + mean content | Learned TT |
|---|---|---:|---:|---:|---:|
| 2 | all | 17550 | 13.95 | 19.08 | 24.11 |
| 2 | head | 9900 | 19.77 | 29.78 | 37.21 |
| 2 | mid | 3381 | 7.48 | 6.60 | 9.17 |
| 2 | tail | 3785 | 5.57 | 4.12 | 5.55 |
| 2 | unseen | 484 | 5.79 | 4.55 | 5.58 |
| 3 | all | 8571 | 40.75 | 56.22 | 55.37 |
| 3 | head | 7387 | 43.43 | 62.23 | 58.58 |
| 3 | mid | 763 | 24.12 | 19.27 | 37.75 |
| 3 | tail | 380 | 22.89 | 17.11 | 31.58 |
| 3 | unseen | 41 | 34.15 | 24.39 | 26.83 |

The learned scorer improves aggregate prefix-2 ranking, primarily head and mid. It does not improve tail/unseen over raw prefix-plus-content scoring: tail hits 210 versus 211, unseen 27 versus 28. This weakens the hypothesis that replacing the mean-profile ranker alone solves the rare-item bottleneck. Depth 3 loses most rare targets before ranking (tail 380, unseen 41 covered).

## Hybrid trade-offs

Five depth-2 TT slots preserve aggregate hits almost exactly: +1 total, with head -39, mid +17, tail +22, unseen +1. Ten slots yield -64 total, head -165, mid +51, tail +45, unseen +5. Five depth-3 TT slots produce +14 total but unseen -1. These small aggregate changes are not evidence of a statistically reliable overall improvement. Top-10/20 are unchanged by construction.

The missing comparison is the same quota hybrid using full-catalog TT candidates without a GenRec prefix restriction. Without it, improvements over exact GenRec cannot be attributed to prefix retrieval rather than adding a second model.

## Why the content Two-Tower result is not contradictory

This repository uses BGE features through a shared learned item MLP, including for interaction-unseen items. It has no per-item trainable ID lookup. The user tower is sequential. Thus it already has content-based inductive capability; it does not need a separate content-only fallback. GenRec uses compressed semantic representations and must score a complete autoregressive SID path. A representable unseen item need not receive sufficient probability to enter the beam or top-K.

[TIGER §4.3](https://arxiv.org/html/2305.05065v3#S4.SS3) evaluates a special cold-start procedure: remove 5% of test items from training, include unseen items matching the first three generated codes, and control their maximum fraction with epsilon. Its cold-start comparator there is Semantic_KNN, not this trained content Two-Tower. This does not establish that ordinary full-SID beam search generally beats content-aware dense retrieval.

[LIGER §§2.3–2.4](https://arxiv.org/html/2411.18814v2) reports cold-start failures of TIGER/TIGER(T) and nonzero dense retrieval under its own protocol, linking the failure to low generated path probabilities. Its hybrid also explicitly adds cold-start items to candidates before dense ranking (§3); those results cannot be credited solely to generative candidate coverage. These are counterexamples to a universal superiority claim, not proof that dense retrieval always wins.

## Decision and limits

Keep the observed positive claim narrow: learned GenRec prefixes cover rare-item targets better with the correct history than with shuffled/popularity controls. The stronger claim that semantic-ID generation improves rare-item final recall over content-aware dense retrieval remains unsupported here.

Next bounded diagnostic: compare exact-GenRec + full-TT quota hybrids against the existing prefix-TT hybrids with identical 5/10 quotas and novelty rules. Record per-user hit/rank outcomes for paired uncertainty. If full-TT hybrids match or beat prefix hybrids, stop claiming an added rare-item ranking benefit from prefix restriction. A trained semantic-vs-permuted-SID ablation is still required to isolate semantic structure itself.

Selection and multiple comparisons use validation; the small best hybrid gain must not be presented as a confirmed test improvement. Existing checkpoint selection objectives differ and the full test set has already been explored. This is a comparison of fixed checkpoints, not an architecture-wide superiority result. Unseen means no training interactions, not a temporal new-catalog-item experiment.

Runtime: 25.88 minutes total; real GenRec 111.97s, shuffled GenRec 108.59s, TT encoding/full scoring 9.66s, CPU ranking and controls 1321.95s. This diagnostic reuses full-catalog scores; these components are not an end-to-end production latency benchmark.
