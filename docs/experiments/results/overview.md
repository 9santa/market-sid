# Saved results

[Notebook index](../README.md) · [Interpretations](../research_log.md) · [Earlier experiments](../archive.md)

Snapshot reviewed 2026-09-19. Tables below are transcribed programmatically from existing JSON artifacts; no model evaluation was rerun. Rates and NDCG are shown as percentages unless marked otherwise. Top-10 hit counts are `round(n × recall@10)`. Source links point to local, generally Git-ignored artifacts; the important values are retained here.

Jump to: [SID comparison](#main-sid-comparison) ·
[Tokenizer exports](#tokenizer-structure-and-export) ·
[Path likelihood](#path-likelihood-and-history-sensitivity) ·
[Decoding interventions](#decoding-interventions) ·
[Other datasets](#separate-benchmarks).

The [baseline Recall@20 ledger](baseline_development.md#retained-run-ledger)
uses fractions and warm targets; it is kept separate from these full-population
generative-retrieval tables. For example, 0.030925 there means 3.0925%.

## Main SID comparison

All five saved evaluations use the main test population (80,471 users), the full retained catalog, and beam 100. GenRec C is fixed. Content uses checkpoint epoch 50; the others use 49. The content result has older, less complete provenance metadata; it lacks the data/config signatures present in the later ablation files. Treat matching population/settings as a useful comparison, not a complete reproducibility guarantee.

| SID source | R@10 % | NDCG@10 % | R@20 % | R@50 % | Non-head top-10 hits |
| --- | --- | --- | --- | --- | --- |
| content | 2.1809 | 1.1416 | 3.3068 | 5.3075 | 26 |
| twotower | 2.3822 | 1.2590 | 3.5976 | 5.6691 | 17 |
| rq4_a | 2.2182 | 1.1925 | 3.3105 | 5.3311 | 17 |
| rq4_b | 2.2107 | 1.1921 | 3.3416 | 5.2255 | 19 |
| rq4_c | 2.2604 | 1.1886 | 3.4000 | 5.3411 | 13 |


`content`: BGE + 3 RQ + collision; `twotower`: Two-Tower + 3 RQ + collision. `rq4_a`: Two-Tower + 4 RQ; `rq4_b`: BGE + final-depth Two-Tower supervision; `rq4_c`: BGE + all-depth Two-Tower supervision. See [the ablation protocol](../protocols/sid_ablation.md).

### content

Source: [genrec_content_C_test_prefix_survival_metrics.json](../../../artifacts/main/evaluation/genrec_content_C_test_prefix_survival_metrics.json).

| Group | n | Prefix 1 % | Prefix 2 % | Prefix 3 % | Full SID % | R@10 % | Top-10 hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 80471 | 84.9076 | 22.5336 | 9.3164 | 6.9864 | 2.1809 | 1755 |
| head | 21296 | 90.1860 | 38.4391 | 26.9675 | 24.6431 | 8.1189 | 1729 |
| mid | 19141 | 84.7709 | 19.2675 | 4.7437 | 1.4733 | 0.1045 | 20 |
| tail | 33742 | 82.3217 | 15.8319 | 2.2109 | 0.2608 | 0.0119 | 4 |
| unseen | 6292 | 81.3255 | 14.5741 | 1.5893 | 0.0636 | 0.0318 | 2 |

### twotower

Source: [genrec_twotower_C_test_prefix_survival_metrics.json](../../../artifacts/main/evaluation/genrec_twotower_C_test_prefix_survival_metrics.json).

| Group | n | Prefix 1 % | Prefix 2 % | Prefix 3 % | Full SID % | R@10 % | Top-10 hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 80471 | 84.2825 | 19.8519 | 9.2518 | 7.5642 | 2.3822 | 1917 |
| head | 21296 | 88.7491 | 39.3501 | 29.5689 | 27.0051 | 8.9219 | 1900 |
| mid | 19141 | 84.8441 | 16.1799 | 3.6884 | 1.3740 | 0.0888 | 17 |
| tail | 33742 | 81.9039 | 11.3716 | 1.1499 | 0.1956 | 0.0000 | 0 |
| unseen | 6292 | 80.2130 | 10.5054 | 0.8582 | 0.1113 | 0.0000 | 0 |

### rq4_a

Source: [genrec_rq4_a_C_test_prefix_survival_metrics.json](../../../artifacts/main/evaluation/genrec_rq4_a_C_test_prefix_survival_metrics.json).

| Group | n | Prefix 1 % | Prefix 2 % | Prefix 3 % | Full SID % | R@10 % | Top-10 hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 80471 | 90.6066 | 22.0228 | 8.9200 | 7.0895 | 2.2182 | 1785 |
| head | 21296 | 94.2665 | 42.4117 | 28.3950 | 25.9298 | 8.3020 | 1768 |
| mid | 19141 | 90.5909 | 17.3136 | 3.4585 | 0.8359 | 0.0888 | 17 |
| tail | 33742 | 88.8448 | 13.5973 | 1.1973 | 0.0682 | 0.0000 | 0 |
| unseen | 6292 | 87.7146 | 12.5238 | 1.0331 | 0.0000 | 0.0000 | 0 |

### rq4_b

Source: [genrec_rq4_b_C_test_prefix_survival_metrics.json](../../../artifacts/main/evaluation/genrec_rq4_b_C_test_prefix_survival_metrics.json).

| Group | n | Prefix 1 % | Prefix 2 % | Prefix 3 % | Full SID % | R@10 % | Top-10 hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 80471 | 100.0000 | 35.8166 | 10.9804 | 6.8435 | 2.2107 | 1779 |
| head | 21296 | 100.0000 | 49.0515 | 29.3482 | 24.8263 | 8.2645 | 1760 |
| mid | 19141 | 100.0000 | 34.0944 | 6.7290 | 0.9195 | 0.0993 | 19 |
| tail | 33742 | 100.0000 | 29.6455 | 3.2837 | 0.1097 | 0.0000 | 0 |
| unseen | 6292 | 100.0000 | 29.3547 | 3.0197 | 0.1113 | 0.0000 | 0 |

### rq4_c

Source: [genrec_rq4_c_C_test_prefix_survival_metrics.json](../../../artifacts/main/evaluation/genrec_rq4_c_C_test_prefix_survival_metrics.json).

| Group | n | Prefix 1 % | Prefix 2 % | Prefix 3 % | Full SID % | R@10 % | Top-10 hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 80471 | 100.0000 | 31.3554 | 9.8445 | 7.0125 | 2.2604 | 1819 |
| head | 21296 | 100.0000 | 44.6891 | 28.5218 | 25.2911 | 8.4805 | 1806 |
| mid | 19141 | 100.0000 | 29.2148 | 5.1930 | 1.0762 | 0.0679 | 13 |
| tail | 33742 | 100.0000 | 25.4253 | 2.2465 | 0.1363 | 0.0000 | 0 |
| unseen | 6292 | 100.0000 | 24.5391 | 1.5257 | 0.0795 | 0.0000 | 0 |

### Conditional survival between positions

These divide successive unconditional survival rates. They describe retained targets, not isolated token classification accuracy. Global beam competition and cumulative scores still matter.

| SID source | Group | 2 given 1 % | 3 given 2 % | 4 given 3 % |
| --- | --- | --- | --- | --- |
| content | head | 42.6221 | 70.1564 | 91.3808 |
| content | mid | 22.7290 | 24.6204 | 31.0573 |
| content | tail | 19.2317 | 13.9648 | 11.7962 |
| content | unseen | 17.9207 | 10.9051 | 4.0000 |
| twotower | head | 44.3386 | 75.1432 | 91.3292 |
| twotower | mid | 19.0702 | 22.7963 | 37.2521 |
| twotower | tail | 13.8841 | 10.1121 | 17.0103 |
| twotower | unseen | 13.0969 | 8.1694 | 12.9630 |
| rq4_a | head | 44.9913 | 66.9508 | 91.3180 |
| rq4_a | mid | 19.1119 | 19.9759 | 24.1692 |
| rq4_a | tail | 15.3046 | 8.8056 | 5.6931 |
| rq4_a | unseen | 14.2779 | 8.2487 | 0.0000 |
| rq4_b | head | 49.0515 | 59.8315 | 84.5920 |
| rq4_b | mid | 34.0944 | 19.7364 | 13.6646 |
| rq4_b | tail | 29.6455 | 11.0767 | 3.3394 |
| rq4_b | unseen | 29.3547 | 10.2870 | 3.6842 |
| rq4_c | head | 44.6891 | 63.8226 | 88.6730 |
| rq4_c | mid | 29.2148 | 17.7754 | 20.7243 |
| rq4_c | tail | 25.4253 | 8.8355 | 6.0686 |
| rq4_c | unseen | 24.5391 | 6.2176 | 5.2083 |

## Tokenizer structure and export

Content/Industrial prefix counts were recomputed from saved SID arrays during this documentation review. Other counts come from the saved evaluation/export metadata. Current-array counts are not a substitute for missing historical SID hashes.

| Tokenizer | Unique c1 | Unique c1,c2 | Unique c1,c2,c3 | Unique full SID | All root branches fit beam 100? |
| --- | --- | --- | --- | --- | --- |
| content | 256 | 29590 | 106572 | 134288 | No |
| twotower | 256 | 40383 | 118943 | 134288 | No |
| rq4_a | 202 | 33151 | 116131 | 134288 | No |
| rq4_b | 73 | 13671 | 98037 | 134288 | Yes |
| rq4_c | 80 | 15307 | 105042 | 134288 | Yes |
| Industrial | 66 | 6466 | 17667 | 25848 | Yes |


Four-RQ export uses unique minimum-cost fourth assignments inside fixed three-code buckets. “Raw collision rate” is `1 − unique full codes / items`, not the fraction of items belonging to a collision bucket. Added squared error is measured in each tokenizer’s own latent space; do not compare its absolute scale across tokenizers as recommendation quality.

| Variant / source | Raw collision % | Largest 3-code bucket | Changed fourth codes | Extra squared error / item | Final collision % |
| --- | --- | --- | --- | --- | --- |
| [A](../../../artifacts/main/tokenizers/rqvae_rq4_a_export.json) | 3.1522 | 49 | 4342 | 0.00020313 | 0.0000 |
| [B](../../../artifacts/main/tokenizers/rqvae_rq4_b_export.json) | 5.8769 | 68 | 8155 | 0.00207553 | 0.0000 |
| [C](../../../artifacts/main/tokenizers/rqvae_rq4_c_export.json) | 4.5306 | 43 | 6243 | 0.00246106 | 0.0000 |

### SVD SID churn

Recomputed from existing exported arrays, relative to saved content semantic codes. This measures assignment changes, not locality or recall. Exact locality output from the earlier discussion was not recovered.

| Behavior weight / source | Changed c1 % | Changed c1,c2 % | Changed c1,c2,c3 % |
| --- | --- | --- | --- |
| [0.001](../../../data/processed/item_semantic_ids_behavior_0p001.npy) | 4.0778 | 31.3580 | 69.1573 |
| [0.005](../../../data/processed/item_semantic_ids_behavior_0p005.npy) | 5.8322 | 39.4473 | 79.2565 |
| [0.01](../../../data/processed/item_semantic_ids_behavior_0p01.npy) | 11.8789 | 53.3354 | 88.3325 |
| [0.05](../../../data/processed/item_semantic_ids_behavior_0p05.npy) | 22.2641 | 70.5968 | 95.8120 |

## Path likelihood and history sensitivity

Source: [genrec_twotower_C_test_path_diagnostics.json](../../../artifacts/main/diagnostics/paths/genrec_twotower_C_test_path_diagnostics.json). Main test, 80,471 targets; teacher forcing, full-vocabulary softmax, no seen filtering. Values are natural log probabilities (higher/less negative is better). Training priors count 574,699 target occurrences; grouping still uses all train interactions.

| Group | n | Mean log P(c1) | Mean log P(c2) | Mean log P(c3) | Mean log P(c4) | Mean full log P |
| --- | --- | --- | --- | --- | --- | --- |
| all | 80471 | -4.6203 | -4.4111 | -3.7687 | -0.3788 | -13.1789 |
| head | 21296 | -4.4778 | -2.9214 | -1.0501 | -0.2239 | -8.6733 |
| mid | 19141 | -4.6523 | -4.5117 | -3.3720 | -0.3695 | -12.9055 |
| tail | 33742 | -4.6666 | -5.1333 | -5.3120 | -0.4498 | -15.5616 |
| unseen | 6292 | -4.7574 | -5.2742 | -5.9006 | -0.5506 | -16.4828 |


Positions 2+3 contribute **93.98%** of the mean head–tail full-path log-probability difference. Position 1 accounts for **87.64%** of the mean real-minus-shuffled benefit. These are descriptive decompositions, not causal attribution.

| Group | Real − shuffled c1 | c2 | c3 | c4 |
| --- | --- | --- | --- | --- |
| all | 1.4526 | 0.1213 | 0.0760 | 0.0074 |
| head | 1.1632 | 0.1431 | 0.1006 | 0.0116 |
| mid | 1.4273 | 0.1063 | 0.0734 | 0.0055 |
| tail | 1.6365 | 0.1118 | 0.0584 | 0.0049 |
| unseen | 1.5234 | 0.1448 | 0.0955 | 0.0126 |


Correlations over all target rows (Spearman). Marginal token counts ignore parent prefixes; conditional priors do not. Conditional-prior correlations were recomputed from the saved Parquet rows; the others are stored in the diagnostic JSON.

| Position | Marginal token count vs token log P | Joint prefix count vs prefix log P | Conditional log prior vs token log P |
| --- | --- | --- | --- |
| 1 | 0.4783 | 0.4783 | 0.4783 |
| 2 | 0.2604 | 0.7737 | 0.9270 |
| 3 | 0.1768 | 0.8147 | 0.4833 |
| 4 | 0.6026 | 0.7712 | 0.6441 |

### Singleton third-level branches

For target rows whose catalog c1,c2 prefix has exactly one c3 child, the sole valid child’s mean raw probability is:

| Group | Target rows | Mean valid mass % |
| --- | --- | --- |
| head | 1813 | 83.7151 |
| mid | 2026 | 17.2204 |
| tail | 4046 | 2.9898 |
| unseen | 817 | 1.4354 |


Computed by joining catalog singleton parents to the saved teacher-forced diagnostic rows. This is a target-frequency-weighted mean, not an equal-parent mean. Reproduction definitions and source hashes: [derived statistics](../../../artifacts/main/derived/derived_statistics.json).

## Decoding interventions

All following runs use main Two-Tower GenRec C, beam 100, and the full catalog. Priors use beta 0.5. Validation has different target-group sizes than test: head 23,867; mid 19,843; tail 32,279; unseen 4,482. Do not pool validation and test rows.

### All-level prior sweep

Validation source: [genrec_twotower_C_valid_prior_sweep.json](../../../artifacts/main/diagnostics/decoding/genrec_twotower_C_valid_prior_sweep.json).

| Alpha | R@10 % | NDCG@10 % | R@50 % | Head hits | Mid hits | Tail hits | Unseen hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 2.8246 | 1.5078 | 6.6185 | 2243 | 23 | 6 | 1 |
| 0.05 | 2.8097 | 1.4956 | 6.5651 | 2229 | 24 | 6 | 2 |
| 0.1 | 2.7612 | 1.4815 | 6.5129 | 2190 | 24 | 6 | 2 |
| 0.2 | 2.6705 | 1.4330 | 6.4122 | 2112 | 26 | 9 | 2 |
| 0.3 | 2.5500 | 1.3660 | 6.1662 | 2014 | 26 | 10 | 2 |
| 0.5 | 2.1548 | 1.1311 | 5.3224 | 1665 | 37 | 21 | 11 |

### c2-only prior sweep

Validation source: [genrec_twotower_C_valid_prior_sweep_levels2.json](../../../artifacts/main/diagnostics/decoding/genrec_twotower_C_valid_prior_sweep_levels2.json).

| Alpha | R@10 % | NDCG@10 % | R@50 % | Head hits | Mid hits | Tail hits | Unseen hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 2.8246 | 1.5078 | 6.6185 | 2243 | 23 | 6 | 1 |
| 0.05 | 2.8047 | 1.4971 | 6.5924 | 2227 | 23 | 6 | 1 |
| 0.1 | 2.7836 | 1.4877 | 6.5763 | 2209 | 24 | 6 | 1 |
| 0.2 | 2.7327 | 1.4630 | 6.4731 | 2168 | 24 | 5 | 2 |
| 0.3 | 2.5910 | 1.3935 | 6.3277 | 2052 | 27 | 4 | 2 |
| 0.5 | 2.2766 | 1.2238 | 5.7313 | 1790 | 33 | 7 | 2 |


Alpha zero has the best aggregate validation R@10 and NDCG@10 in both grids. Nonzero corrections are trade-off experiments, not aggregate winners.

### Normalization and correction summaries

| Split | Scoring | R@10 % | NDCG@10 % | R@50 % | All top-10 hits | Non-head top-10 hits |
| --- | --- | --- | --- | --- | --- | --- |
| valid | Full vocabulary, alpha 0 | 2.8246 | 1.5078 | 6.6185 | 2273 | 30 |
| valid | Trie normalized, alpha 0 | 2.7936 | 1.4961 | 6.5328 | 2248 | 44 |
| test | Full vocabulary, alpha 0 | 2.3822 | 1.2590 | 5.6691 | 1917 | 17 |
| test | Trie normalized, alpha 0 | 2.3822 | 1.2537 | 5.5970 | 1917 | 39 |
| test | Full vocabulary, alpha 0.5 all levels | 1.8317 | 0.9704 | 4.6402 | 1474 | 53 |


Sources: [genrec_twotower_C_valid_prefix_survival_trie_normalized_metrics.json](../../../artifacts/main/evaluation/genrec_twotower_C_valid_prefix_survival_trie_normalized_metrics.json), [genrec_twotower_C_test_prefix_survival_trie_normalized_metrics.json](../../../artifacts/main/evaluation/genrec_twotower_C_test_prefix_survival_trie_normalized_metrics.json), [genrec_twotower_C_test_prior_alpha0.5.json](../../../artifacts/main/diagnostics/decoding/genrec_twotower_C_test_prior_alpha0.5.json). Baselines are the validation alpha-zero sweep row and the test Two-Tower prefix-survival file.


Test group comparison (R@10 in percent; counts in parentheses):

| Group | Full vocabulary | Trie normalized | Prior alpha 0.5 |
| --- | --- | --- | --- |
| head | 8.9219 (1900) | 8.8186 (1878) | 6.6726 (1421) |
| mid | 0.0888 (17) | 0.1672 (32) | 0.1724 (33) |
| tail | 0.0000 (0) | 0.0148 (5) | 0.0326 (11) |
| unseen | 0.0000 (0) | 0.0318 (2) | 0.1430 (9) |


Trie-normalized test prefix survival:

| Group | n | Prefix 1 % | Prefix 2 % | Prefix 3 % | Full SID % | R@10 % | Top-10 hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 80471 | 84.2825 | 19.8009 | 9.1959 | 7.4524 | 2.3822 | 1917 |
| head | 21296 | 88.7491 | 39.2233 | 28.6814 | 26.0894 | 8.8186 | 1878 |
| mid | 19141 | 84.8441 | 16.1225 | 4.1429 | 1.7136 | 0.1672 | 32 |
| tail | 33742 | 81.9039 | 11.3538 | 1.3188 | 0.2964 | 0.0148 | 5 |
| unseen | 6292 | 80.2130 | 10.5531 | 0.8582 | 0.2066 | 0.0318 | 2 |


Validation aggregate metrics slightly favor full-vocabulary normalization. Equal test R@10 does not mean identical rankings: 22 fewer head hits are balanced by 22 additional non-head hits. Test NDCG@10 and R@50 decline slightly. No confidence intervals were saved.

## Separate benchmarks

These datasets are separate populations with their own tokenizers. They are not rows of a main-dataset model ablation. Smoke runs are listed only to preserve implementation history.

| Dataset / source | Variant | Epoch | Users | Beam | Scope | R@10 % | NDCG@10 % | R@50 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [movielens-1m](../../../artifacts/benchmarks/movielens-1m/evaluation/genrec_A_best_test_metrics.json) | A | 12 | 6040 | 100 | full | 28.1623 | 16.5430 | 52.2185 |
| [amazon-industrial-scientific](../../../artifacts/benchmarks/amazon-industrial-scientific/evaluation/genrec_C_best_test_prefix_survival_metrics.json) | C | 50 | 50984 | 100 | full | 3.6972 | 1.8936 | 9.6187 |
| [amazon-industrial-scientific](../../../artifacts/benchmarks/amazon-industrial-scientific/evaluation/genrec_C_best_test_metrics.json) | C | 50 | 50984 | 500 | full | 3.6972 | 1.8936 | 9.6364 |
| [amazon-beauty](../../../artifacts/benchmarks/amazon-beauty/evaluation/genrec_C_smoke_best_test_limit10_metrics.json) | C | 1 | 10 | 200 | smoke | 0.0000 | 0.0000 | 0.0000 |

### Industrial prefix survival

This is the original 50,984-user diagnostic discussed in the project. Source: [genrec_C_best_test_prefix_survival_metrics.json](../../../artifacts/benchmarks/amazon-industrial-scientific/evaluation/genrec_C_best_test_prefix_survival_metrics.json).

| Group | n | Prefix 1 % | Prefix 2 % | Prefix 3 % | Full SID % | R@10 % | Top-10 hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 50984 | 100.0000 | 35.4896 | 19.2982 | 13.1826 | 3.6972 | 1885 |
| head | 20594 | 100.0000 | 49.4901 | 35.1704 | 29.9747 | 8.9881 | 1851 |
| mid | 18691 | 100.0000 | 28.4843 | 9.9513 | 2.5039 | 0.1819 | 34 |
| tail | 11367 | 100.0000 | 21.7648 | 6.1846 | 0.6510 | 0.0000 | 0 |
| unseen | 332 | 100.0000 | 31.3253 | 9.9398 | 1.8072 | 0.0000 | 0 |

| Group | 2 given 1 % | 3 given 2 % | 4 given 3 % |
| --- | --- | --- | --- |
| head | 49.4901 | 71.0655 | 85.2271 |
| mid | 28.4843 | 34.9361 | 25.1613 |
| tail | 21.7648 | 28.4155 | 10.5263 |
| unseen | 31.3253 | 31.7308 | 18.1818 |


The current saved Industrial SID array has 25,848 items and 66 root codes; beam 100 retains every first-level branch. MovieLens has only 11 tail and 2 unseen test targets under these thresholds. Beauty’s 10-user zero-hit smoke output is not evidence about converged quality.
