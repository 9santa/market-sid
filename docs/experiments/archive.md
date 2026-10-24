# Earlier and archived experiments

[Notebook index](README.md) · [Research log](research_log.md) · [Main results](results/overview.md)

Older experiments and side branches are collected here. Some runners are no longer available; original artifact names are kept. Metrics are percentages.

## Baseline hyperparameter runs

The [baseline development notes](results/baseline_development.md) cover early debugging and Recall@20 results. This page mostly uses R@10.

The baseline histories are dated 2026-09-08. Both models were selected using validation warm-target Recall@20. A recorded finish and test metric mark a run as complete but do not guarantee all planned epochs ran. Missing training settings remain unspecified.

“Test all” includes all target groups but not necessarily all catalog candidates. Older SASRec runs do not log candidate IDs; the current runner uses warm candidates. Two-Tower saves warm and full-catalog metrics, and the table uses the full catalog. These are not candidate-matched leaderboard scores.

Across these saved configurations, SASRec learning rate is 0.001 and Two-Tower
learning rate is 0.0003; both use dropout 0.2, weight decay 0.0001, and seed 42.
Full recorded configurations are preserved in the evidence manifest.

### SASRec

| Run UTC time / source | Dim | Layers | History | Batch | Training strategy | State | Best epoch | Valid warm R@20 % | Test all R@10 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [085159839650](../../artifacts/main/training/sasrec_history_20260908T085159839650Z.json) | 64 | 4 | 50 | 1024 | unspecified | complete | 9 | 2.7767 | 1.3980 |
| [085655994343](../../artifacts/main/training/sasrec_history_20260908T085655994343Z.json) | 128 | 4 | 50 | 1024 | unspecified | complete | 6 | 2.5622 | 1.3595 |
| [090154248919](../../artifacts/main/training/sasrec_history_20260908T090154248919Z.json) | 256 | 4 | 50 | 1024 | unspecified | complete | 6 | 2.5333 | 1.3980 |
| [135021287418](../../artifacts/main/training/sasrec_history_20260908T135021287418Z.json) | 64 | 2 | 400 | 2048 | unspecified | incomplete | None | — | — |
| [135038969390](../../artifacts/main/training/sasrec_history_20260908T135038969390Z.json) | 64 | 2 | 400 | 1024 | unspecified | incomplete | None | — | — |
| [135049322512](../../artifacts/main/training/sasrec_history_20260908T135049322512Z.json) | 64 | 2 | 200 | 1024 | unspecified | incomplete | None | — | — |
| [135250796248](../../artifacts/main/training/sasrec_history_20260908T135250796248Z.json) | 64 | 2 | 50 | 1024 | unspecified | incomplete | None | — | — |
| [143126184389](../../artifacts/main/training/sasrec_history_20260908T143126184389Z.json) | 64 | 2 | 200 | 1024 | all_prefixes | complete | 3 | 3.0254 | 1.6341 |
| [144322654352](../../artifacts/main/training/sasrec_history_20260908T144322654352Z.json) | 64 | 3 | 50 | 1024 | all_prefixes | complete | 4 | 3.0689 | 1.6031 |

### Two-Tower

| Run UTC time / source | Dim | Layers | History | Batch | Training strategy | State | Best epoch | Valid warm R@20 % | Test all R@10 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [122822582355](../../artifacts/main/training/two_tower_history_20260908T122822582355Z.json) | 128 | 2 | 50 | 1024 | unspecified | smoke | 1 | 0.2058 | 0.1953 |
| [124156726516](../../artifacts/main/training/two_tower_history_20260908T124156726516Z.json) | 128 | 2 | 50 | 1024 | unspecified | complete | 19 | 3.4150 | 1.8516 |
| [125021681353](../../artifacts/main/training/two_tower_history_20260908T125021681353Z.json) | 128 | 2 | 50 | 1024 | all_prefixes | smoke | 1 | 0.2058 | 0.0000 |
| [125126375320](../../artifacts/main/training/two_tower_history_20260908T125126375320Z.json) | 128 | 2 | 50 | 1024 | all_prefixes | incomplete | 2 | 2.8570 | — |
| [125631038709](../../artifacts/main/training/two_tower_history_20260908T125631038709Z.json) | 128 | 2 | 50 | 2048 | all_prefixes | complete | 28 | 3.7466 | 1.8752 |
| [131000075748](../../artifacts/main/training/two_tower_history_20260908T131000075748Z.json) | 256 | 2 | 50 | 2048 | all_prefixes | incomplete | 7 | 2.3016 | — |
| [131458161044](../../artifacts/main/training/two_tower_history_20260908T131458161044Z.json) | 64 | 2 | 50 | 2048 | all_prefixes | complete | 6 | 3.2189 | 1.6428 |
| [131904621494](../../artifacts/main/training/two_tower_history_20260908T131904621494Z.json) | 128 | 3 | 50 | 2048 | all_prefixes | complete | 35 | 3.4558 | 1.8640 |

SASRec dimension 64/128/256 runs do not show a monotonic benefit. Later all-prefix runs improve the recorded validation selection score, but architecture and history settings also differ.

Two-Tower’s 128-dimensional, two-layer, batch-2048 run has the highest saved validation selection score among its completed non-smoke histories. The exported teacher, however, is the later three-layer, epoch-35 checkpoint, as recorded in [item_two_tower_embeddings.json](../../data/processed/item_two_tower_embeddings.json).

## Earlier GenRec histories and limited evaluations

The four `genrec_history_20260910...` files predate explicit SID-source/variant fields. Three record completed 50-epoch runs and one has no epochs. They are not enough to reconstruct an attributable GenRec A/B/C/D comparison. Later Two-Tower/RQ4 histories also include empty or interrupted starts; best epoch 1 in an interrupted history is not the final trained model. The evidence manifest retains all these files.

The following older evaluation files differ in population, checkpoint, or provenance. Do not substitute them for the named main SID comparison.

| Source | Epoch | Split | Users | Beam | R@10 % | R@50 % |
| --- | --- | --- | --- | --- | --- | --- |
| [genrec_test_limit10000_metrics.json](../../artifacts/archive/legacy_main/genrec_test_limit10000_metrics.json) | 50 | test | 10000 | 100 | 2.1500 | 5.0500 |
| [genrec_test_limit1024_metrics.json](../../artifacts/archive/legacy_main/genrec_test_limit1024_metrics.json) | 50 | test | 1024 | 100 | 1.4648 | 4.6875 |
| [genrec_test_limit10_metrics.json](../../artifacts/main/smoke/genrec_test_limit10_metrics.json) | 20 | test | 10 | 200 | 0.0000 | 0.0000 |
| [genrec_test_metrics.json](../../artifacts/archive/legacy_main/genrec_test_metrics.json) | 50 | test | 80471 | 100 | 1.4266 | 3.6100 |
| [genrec_twotower_C_test_metrics.json](../../artifacts/main/evaluation/genrec_twotower_C_test_metrics.json) | 49 | test | 80471 | 100 | 2.3797 | 5.6902 |
| [genrec_valid_limit10_metrics.json](../../artifacts/main/smoke/genrec_valid_limit10_metrics.json) | 20 | valid | 10 | 200 | 0.0000 | 0.0000 |



## Exact ranking and decoder performance

[genrec_diagnosis.json](../../artifacts/archive/legacy_main/genrec_diagnosis.json) contains teacher-forced statistics for 1,024 users and exact catalog ranks for 16 selected users (four per popularity group). Across those 16, exact top-50 lists are 99% head, and beam-100 overlap with exact top 50 averages 85.125%. Tail target exact ranks range from 12,378 to 71,828. This supports a score-distribution problem in that sample, beyond approximate search; it does not establish a full-population exact-retrieval comparison. The file lacks checkpoint/split metadata sufficient to promote it to the main baseline.

[decoder_benchmark.json](../../artifacts/archive/genrec_experiments/decoder_benchmark.json) records identical rankings on eight users, maximum score difference 3.70e-6, and 1.1224 s versus 0.0144 s (77.96×) for an old/new decoder comparison. Another 128-user timing sweep increases throughput from 299 to 900 users/s as the recorded decode batch setting increases from 128 to 4,096.

## Fine-tuning and content-adapted trials

These runs use `normalize_valid: true`, batch 512, and selection by validation NDCG@10 on saved subsets. They change training/checkpoint state, unlike the recent same-checkpoint `--trie-normalize` decoding ablation. Their runner code and exact meaning of historical `normalize_valid`, `sampling`, and `content_features` are not present in the current scripts, so labels are quoted rather than reverse-engineered into unsupported methods.

| Run | Valid users | LR | Sampling label | Content features | Best epoch | Initial NDCG@10 % | Best NDCG@10 % | Best R@10 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [finetune](../../artifacts/archive/genrec_experiments/finetune.json) | 2048 | 0.0001 | unspecified | unspecified | 4 | 1.3550 | 1.4951 | 2.7832 |
| [content_adapted](../../artifacts/archive/genrec_experiments/content_adapted.json) | 2048 | 0.0003 | unspecified | True | 0 | 1.4951 | 1.4951 | 2.7832 |
| [balanced](../../artifacts/archive/genrec_experiments/balanced.json) | 8192 | 0.0001 | recent | False | 5 | 1.4981 | 1.5628 | 2.8931 |
| [balanced_extended](../../artifacts/archive/genrec_experiments/balanced_extended.json) | 8192 | 0.0001 | recent | False | 10 | 1.5628 | 1.6258 | 3.0762 |

`content_adapted` selects epoch 0: no recorded trained epoch improved the selection metric. `balanced`/`balanced_extended` contain `sampling: recent`; their names do not establish inverse-popularity balancing. The extension improves NDCG while validation loss moves from about 2.6318 to 2.6349, illustrating why token loss and retrieval selection can disagree. Different subset sizes make between-row deltas unsuitable as controlled comparisons.

## Collaborative and directional retrieval trials

Saved records add 100 train-only neighboring candidates to beam 200 and tune a `prior_weight` on 8,192 validation users. This is candidate augmentation plus a recorded retrieval score, **not** the later smoothed SID transition-prior subtraction experiment. The exact historical memory/scoring runner is missing. Selected weight is 4 in each recorded grid; neural checkpoints differ between grids.

| Trial | Checkpoint label | Selected weight | Weight-0 NDCG@10 % | Best NDCG@10 % | Best R@10 % |
| --- | --- | --- | --- | --- | --- |
| [retrieval_tuning](../../artifacts/archive/genrec_experiments/retrieval_tuning.json) | checkpoints/genrec_improved_best.pt | 4.0 | 1.4981 | 1.6888 | 3.0396 |
| [directional_tuning](../../artifacts/archive/genrec_experiments/directional_tuning.json) | checkpoints/genrec_improved_best.pt | 4.0 | 1.4981 | 1.6891 | 3.0273 |
| [final_tuning](../../artifacts/archive/genrec_experiments/final_tuning.json) | checkpoints/genrec_best.pt | 4.0 | 1.6256 | 1.8110 | 3.3081 |

Settings and memory signatures also survive in `artifacts/archive/retrieval/genrec_retrieval.json`, `genrec_directional_retrieval.json`, and the corresponding memory metadata. A recommendation-list artifact and smoke settings exist, but qualitative examples alone are not an efficacy result.

## RecBole and saved full/subset comparisons

The [environment artifact](../../artifacts/archive/recbole_benchmark/environment.json) records RecBole 1.2.1 and source/checkpoint hashes. The [training history](../../artifacts/archive/recbole_benchmark/training.json) records an all-prefix SASRec implementation with full-catalog cross-entropy, dimension 64, three layers, and best epoch 4. The evaluation below records checkpoint epoch **3**; this provenance discrepancy is unresolved, so the table preserves the evaluation's value. This is not an isolated library substitution: its loss differs from the repository baseline with sampled negatives. The dedicated training/evaluation runner is not present in the current checkout.

| Model label / source | Test users | Epoch | R@10 % | NDCG@10 % | R@50 % |
| --- | --- | --- | --- | --- | --- |
| [original_sasrec](../../artifacts/archive/recbole_benchmark/original_sasrec_test.json) | 80471 | 4 | 1.6031 | 0.8418 | 4.2363 |
| [recbole_sasrec](../../artifacts/archive/recbole_benchmark/recbole_sasrec_test.json) | 80471 | 3 | 1.9883 | 1.0599 | 4.7769 |
| [original_genrec](../../artifacts/archive/recbole_benchmark/original_genrec_test.json) | 80471 | 20 | 1.9535 | 0.9728 | 4.9074 |
| [improved_genrec_neural](../../artifacts/archive/recbole_benchmark/improved_genrec_neural_test.json) | 10000 | 10 | 2.4100 | 1.2746 | 6.0600 |
| [improved_genrec](../../artifacts/archive/recbole_benchmark/improved_genrec_test.json) | 10000 | 10 | 2.4400 | 1.3271 | 6.3200 |

All rows record the same main mapping/sequence fingerprints, but the last two use **10,000 users**, not 80,471. Their saved user indices match each other; thus the neural-versus-augmented comparison is paired on that subset. They cannot be directly ranked against the full-population rows above. The original GenRec row uses an epoch-20 checkpoint and beam 200, not the later epoch-50 content-C baseline.

The augmented subset result is 2.44% R@10 versus 2.41% for its neural checkpoint, with NDCG@10 1.3271% versus 1.2746%. This is a small measured improvement for that recorded configuration, not proof of a general retrieval method or a pure generative result.

## Evidence gaps and ideas without completed comparisons

- Stronger BGE/alternative encoder and longer text context was considered.
- Explicit SASRec-teacher tokenization. The completed four-RQ teacher is Two-Tower.
- Depth weights such as 1, 0.5, 0.25 were considered; implemented four-RQ C uses an equal mean.
- Larger final-only codebook (512/1,024) considered for oversized buckets; the completed runs use 256 and fit.
- Seen-aware trie constraints: considered; current standard evaluation filters *after* beam search.
- A fifth arbitrary collision token: considered as an alternative, not the selected four-RQ implementation.
- SVD locality figures: discussion reports no improvement, but printed values were not recovered.
- Beauty: smoke metrics only in this workspace; no recovered full-quality result.

No new training command is invented for an archived runner that is absent. Use the current runnable protocols only for the current implementations.
