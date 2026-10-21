# Learned SID assignment control — 2026-10-18

Protocol: [original versus permuted SIDs](../protocols/sid_permutation_control.md).
Generated comparison: [validation table](../../../artifacts/main/reports/sid_permutation_valid.md).

## Verified comparison

Both GenRec C models completed 50 epochs on 574,699 training examples with the same configuration, training seed 42, initial trainable-state hash, RQ checkpoint hash, dataset hashes, and 80,471 ordered validation users. Both use beam 100 and the same full catalog. Early stopping was disabled; selection remained lowest teacher-forced validation loss. Selected epochs: original 49, permuted 50.

The actual saved permutation is bijective and the exported array equals original[permutation]. All SID/permutation hashes match evaluation metadata. Both retain [256, 40383, 118943, 134288] distinct prefixes at depths 1–4. The permutation moved 134,260 of 134,288 items (99.979%); fixed points/singletons are recorded. The original source SID hash equals the pre-control Two-Tower SID hash.

## Item-level results

Percent, with Recall@50 hit counts. Zeros below are exact, not rounding.

| Group | n | Original R@10 | Permuted R@10 | Original R@50 (hits) | Permuted R@50 (hits) |
|---|---:|---:|---:|---:|---:|
| all | 80,471 | 2.826 | 1.492 | 6.607 (5317) | 3.453 (2779) |
| head | 23,867 | 9.402 | 5.032 | 21.637 (5164) | 11.644 (2779) |
| mid | 19,843 | 0.116 | 0.000 | 0.575 (114) | 0.000 (0) |
| tail | 32,279 | 0.019 | 0.000 | 0.105 (34) | 0.000 (0) |
| unseen | 4,482 | 0.022 | 0.000 | 0.112 (5) | 0.000 (0) |

Overall R@50 falls by 47.73% relative (6.607% to 3.453%); original is 1.913 times permuted. NDCG@50 falls from 2.325% to 1.202%. All 2,779 top-50 hits in the permuted run are head targets. This does not mean every recommended item is head; the artifact describes successful target hits, not exposure composition.

The original gains 2,538 hits over permuted; 2,385 of those are head (about 94%). Relative rare-item effects are severe, but the aggregate improvement remains dominated by head. Unseen item-level evidence is only 5 versus 0 hits and must not be presented as robust general cold-start success.

## Where the difference appears

Each cell is original → permuted, percent of all targets in that group.

| Group | Prefix 1 | Prefix 2 | Prefix 3 | Full SID |
|---|---:|---:|---:|---:|
| all | 84.35 → 63.78 | 21.78 → 11.51 | 10.64 → 5.61 | 8.85 → 4.74 |
| head | 89.01 → 78.66 | 41.45 → 28.96 | 30.91 → 18.08 | 28.25 → 15.98 |
| mid | 83.70 → 62.63 | 16.97 → 6.37 | 3.83 → 0.74 | 1.47 → 0.00 |
| tail | 81.73 → 55.25 | 11.74 → 2.96 | 1.19 → 0.15 | 0.24 → 0.00 |
| unseen | 81.30 → 51.12 | 10.66 → 2.92 | 0.98 → 0.18 | 0.20 → 0.00 |

Tail prefix-2 survival is 3,790 versus 956 targets (3.96x); unseen is 478 versus 131 (3.65x). At prefix 3, original retains 759 mid, 383 tail, and 44 unseen targets; permuted retains 146, 48, and 8. None of the permuted non-head targets survives the final beam step. Thus the difference starts before the collision suffix and is not only final item ranking.

Final pruning compares cumulative scores across paths globally. These losses do not independently prove a faulty collision token. Permutation changes both history representations and target addresses, so this experiment does not isolate input encoding from output routing.

## Teacher-forced accuracy and optimization

Accuracy at each selected checkpoint, percent. Later tokens are conditioned on the correct target prefix during teacher forcing, not a generated prefix.

| Model | Epoch | Validation loss | c1 accuracy | c2 accuracy | c3 accuracy | c4 accuracy |
|---|---:|---:|---:|---:|---:|---:|
| original | 49 | 3.2118 | 9.86 | 14.91 | 30.70 | 88.05 |
| permuted | 50 | 3.3956 | 2.05 | 12.94 | 32.51 | 89.38 |

A train-only constant-c1 baseline chooses token 133 for both mappings. Its validation accuracy is 3.6684% under original assignments and 2.0641% under permuted assignments. The permuted model reaches 2.0542%, essentially the constant baseline; original reaches 9.8619%. This is consistent with much weaker history-conditioned first-token prediction after permutation. Aggregate accuracy does not prove that every prediction is token 133 or that the model ignores history entirely.

Permuted c3/c4 teacher-forced accuracies exceed original despite much worse retrieval. This is not contradictory: supplying the true earlier tokens bypasses failures to find that prefix, and changes the prediction task. Teacher-forced token accuracy alone is not a measure of successful item retrieval.

Both losses still decrease late in training; permuted best is the final epoch. The result is for an equal 50-epoch budget, not proof of equal convergence or the best attainable performance after unlimited tuning.

## Remaining confounds

Permutation bins preserve exact train interaction counts through 20, then use doubling bins for head items; unseen is separate. Prefix frequency distributions are not identical. Total variation between original and permuted training distributions:

| Depth | Interaction-frequency TV | Actual next-item-target-frequency TV |
|---|---:|---:|
| 1 | 0.99% | 2.27% |
| 2 | 2.69% | 6.77% |
| 3 | 3.42% | 9.69% |
| 4 | 3.55% | 10.20% |

The control changes correspondence with the same RQ codebook initialization, which is part of the intended intervention. It measures the utility of learned assignments in GenRec C, not hierarchy independently of initialization. Two-Tower SIDs combine content and learned behavioral information; this is not an isolated test of pure BGE semantics. One permutation seed and one training seed were used, and retrieval was evaluated only for the validation-selected checkpoint.

## Interpretation and decision

The learned item-to-SID assignment has a large positive effect versus frequency-stratified permutation under the matched budget. This is stronger evidence for assignment utility than changing among several learned RQ variants. The original broad claim should nevertheless be split: learned assignments help GenRec compared with scrambled addresses, but this does not imply GenRec beats content-aware dense retrieval on rare items.

The prior full-TT control still stands: prefix restriction did not add rare-item value to the tested quota hybrid. Positive coarse routing, positive learned-assignment utility, and weaker rare-item item-level retrieval than Content Two-Tower can all hold simultaneously.

Freeze this comparison and evaluate both selected checkpoints on test with the same beam/filter settings; use the existing report command with --split test. Existing test-set exploration must be disclosed. If the project budget permits additional replication, repeat the matched pair with predeclared seeds and archive artifacts before overwriting fixed source names. Do not start another tokenizer family in response to this positive result.

No training, model inference, or checkpoint mutation was performed in this analysis. Read-only array checks, train-frequency calculations, and saved history/metric inspection were used.

## Evidence hashes

- artifacts/main/evaluation/genrec_twotower_original_C_valid_prefix_survival_metrics.json: `3e8bc54fbc4dad0108b7e6da7030e2b485e40eebc9a4b6730137aa78dbbf76ea`
- artifacts/main/training/genrec_twotower_original_C_history_20261018T081550870299Z.json: `522da81f4bf20ccf33596cbdc9c418f547aec3e8d7b3922b65026a6e88175b30`
- artifacts/main/evaluation/genrec_twotower_permuted_C_valid_prefix_survival_metrics.json: `cb4f12be6f1e2a37f9a85def737d924638e298d7779716110145b65907b955eb`
- artifacts/main/training/genrec_twotower_permuted_C_history_20261018T084448212680Z.json: `0759e801f1a514eb7943b4e4327c66eb2c782c823a11f5ab72be514e63827764`

Shared initialization: `4a5478d58ba191a382676e234033c5b10065a4dcbd3bce84cba369465f6b4cda`.
Shared RQ checkpoint: `d542c49ca46d74f5c1910ce518fedaf387c14ab4d2c49bed24f3c8bc68dfa2d9`.
