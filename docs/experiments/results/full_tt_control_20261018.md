# Full-catalog Two-Tower hybrid control — 2026-10-18

Source: [validation artifact](../../../artifacts/main/diagnostics/prefix_retrieval/prefix_retrieval_valid_full_tt_control.json).
Per-user paired checks: [derived statistics](../../../artifacts/main/derived/full_tt_control_20261018_statistics.json).

Artifact SHA-256: `41dbe407418e6f0d64a3ff313a00226becdcfa9460d0e095a7b65b1d0b0d871e`.

## Question and protocol

Does restricting novel Two-Tower additions to GenRec prefixes improve the hybrid over adding novel items directly from full-catalog Two-Tower? Same frozen checkpoints, 80,471 validation users, beam 100, depths 2/3, quotas 5/10. All hybrids keep exact top-45/top-40 and exclude the entire exact top-50 from additions. Full TT uses the same learned scores without prefix restriction. No models were trained or reevaluated during this analysis.

The NPZ SHA-256 and ordered-user hash match the JSON. All 26 methods' Recall/NDCG metrics were reconstructed from saved per-user target ranks. Checkpoint/SID/data/user hashes match the prior full-validation run, and all previously available ranking results match exactly. This isolates the newly added full-TT hybrid control.

## Recall@50

Percent, with absolute target hits in parentheses.

| Method | All | Head | Mid | Tail | Unseen |
|---|---:|---:|---:|---:|---:|
| exact | 6.619 (5326) | 21.645 (5166) | 0.595 (118) | 0.112 (36) | 0.134 (6) |
| two_tower_full | 5.143 (4139) | 14.794 (3531) | 1.577 (313) | 0.768 (248) | 1.049 (47) |
| hybrid_full_two_tower_slots5 | 6.658 (5358) | 21.561 (5146) | 0.695 (138) | 0.195 (63) | 0.245 (11) |
| hybrid2_two_tower_slots5 | 6.620 (5327) | 21.482 (5127) | 0.680 (135) | 0.180 (58) | 0.156 (7) |
| hybrid3_two_tower_slots5 | 6.636 (5340) | 21.574 (5149) | 0.685 (136) | 0.155 (50) | 0.112 (5) |
| hybrid_full_two_tower_slots10 | 6.590 (5303) | 21.054 (5025) | 0.842 (167) | 0.304 (98) | 0.290 (13) |
| hybrid2_two_tower_slots10 | 6.539 (5262) | 20.954 (5001) | 0.852 (169) | 0.251 (81) | 0.245 (11) |
| hybrid3_two_tower_slots10 | 6.590 (5303) | 21.243 (5070) | 0.816 (162) | 0.204 (66) | 0.112 (5) |

Full TT additions have the highest observed aggregate recall at five slots. Compared with five-slot prefix-2 they gain 31 hits; compared with prefix-3 they gain 18. At ten slots they gain 41 over prefix-2 and tie prefix-3 overall. Neither prefix-restricted hybrid exceeds full TT additions on tail or unseen for either quota.

The ten-slot prefix-3 tie hides a redistribution: relative to full-TT slots10, head gains 45, mid loses 5, tail loses 32, unseen loses 8. Equal aggregate recall does not mean equal group behavior.

## Paired uncertainty

All deltas and intervals below are percentage points. Intervals are the evaluator's approximate pointwise 95% normal intervals for paired user differences. Exact p-values are two-sided binomial/McNemar tests on discordant hit outcomes. All comparisons are exploratory and unadjusted for multiple comparisons; neither intervals nor tests account for training-seed variation.

| Method minus reference | Group | Net hits | Method-only / reference-only | Delta R@50, pp | Approx. 95% CI, pp | Exact p, unadjusted |
|---|---|---:|---:|---:|---|---:|
| hybrid_full_two_tower_slots5 − exact | all | +32 | 283 / 251 | +0.0398 | [-0.0165, +0.0960] | 0.1797 |
| hybrid_full_two_tower_slots5 − exact | head | -20 | 211 / 231 | -0.0838 | [-0.2564, +0.0889] | 0.3662 |
| hybrid_full_two_tower_slots5 − exact | mid | +20 | 32 / 12 | +0.1008 | [+0.0353, +0.1663] | 0.003658 |
| hybrid_full_two_tower_slots5 − exact | tail | +27 | 32 / 5 | +0.0836 | [+0.0467, +0.1206] | 7.428e-06 |
| hybrid_full_two_tower_slots5 − exact | unseen | +5 | 8 / 3 | +0.1116 | [-0.0335, +0.2566] | 0.2266 |
| hybrid2_two_tower_slots5 − hybrid_full_two_tower_slots5 | all | -31 | 49 / 80 | -0.0385 | [-0.0662, -0.0109] | 0.008008 |
| hybrid2_two_tower_slots5 − hybrid_full_two_tower_slots5 | tail | -5 | 12 / 17 | -0.0155 | [-0.0482, +0.0172] | 0.4583 |
| hybrid2_two_tower_slots5 − hybrid_full_two_tower_slots5 | unseen | -4 | 0 / 4 | -0.0892 | [-0.1767, -0.0018] | 0.125 |
| hybrid3_two_tower_slots5 − hybrid_full_two_tower_slots5 | all | -18 | 127 / 145 | -0.0224 | [-0.0625, +0.0178] | 0.3026 |
| hybrid3_two_tower_slots5 − hybrid_full_two_tower_slots5 | tail | -13 | 16 / 29 | -0.0403 | [-0.0810, +0.0005] | 0.07245 |
| hybrid3_two_tower_slots5 − hybrid_full_two_tower_slots5 | unseen | -6 | 0 / 6 | -0.1339 | [-0.2409, -0.0268] | 0.03125 |
| hybrid_full_two_tower_slots10 − exact | all | -23 | 489 / 512 | -0.0286 | [-0.1056, +0.0485] | 0.4869 |
| hybrid_full_two_tower_slots10 − exact | head | -141 | 332 / 473 | -0.5908 | [-0.8237, -0.3579] | 7.559e-07 |
| hybrid_full_two_tower_slots10 − exact | mid | +49 | 75 / 26 | +0.2469 | [+0.1477, +0.3461] | 1.115e-06 |
| hybrid_full_two_tower_slots10 − exact | tail | +62 | 72 / 10 | +0.1921 | [+0.1371, +0.2470] | 1.022e-12 |
| hybrid_full_two_tower_slots10 − exact | unseen | +7 | 10 / 3 | +0.1562 | [-0.0014, +0.3138] | 0.09229 |
| hybrid2_two_tower_slots10 − hybrid_full_two_tower_slots10 | all | -41 | 95 / 136 | -0.0510 | [-0.0880, -0.0139] | 0.008352 |
| hybrid2_two_tower_slots10 − hybrid_full_two_tower_slots10 | tail | -17 | 16 / 33 | -0.0527 | [-0.0952, -0.0102] | 0.02129 |
| hybrid2_two_tower_slots10 − hybrid_full_two_tower_slots10 | unseen | -2 | 4 / 6 | -0.0446 | [-0.1829, +0.0937] | 0.7539 |
| hybrid3_two_tower_slots10 − hybrid_full_two_tower_slots10 | all | +0 | 269 / 269 | +0.0000 | [-0.0565, +0.0565] | 1 |
| hybrid3_two_tower_slots10 − hybrid_full_two_tower_slots10 | tail | -32 | 31 / 63 | -0.0991 | [-0.1580, -0.0403] | 0.001261 |
| hybrid3_two_tower_slots10 − hybrid_full_two_tower_slots10 | unseen | -8 | 0 / 8 | -0.1785 | [-0.3021, -0.0549] | 0.007812 |

Full-TT slots5 versus exact has 283 wins and 251 losses: net +32, delta +0.0398 pp, approximate CI [-0.0165, +0.0961] pp; exact p=0.180. This is not a confirmed aggregate improvement. Mid and tail show stronger directional evidence; unseen remains sparse (8 wins/3 losses, p=0.227).

Full-TT slots10 versus exact has net -23 overall, exchanging 141 head hits for 49 mid, 62 tail and 7 unseen hits. This is a head/rare trade-off, not an aggregate-quality breakthrough.

The sparse unseen intervals need particular care. Prefix-2 slots5 versus full TT has 0 wins and 4 losses. Its normal CI narrowly excludes zero, but exact p=0.125. Do not call that a reliable unseen improvement/degradation based on the normal interval alone. Prefix-3 slots10 has 0 wins and 8 losses (unadjusted p=0.0078125), a larger directional signal, still within an exploratory family of comparisons.

## What remains of the prefix hypothesis

The previous shuffled-history/popularity controls still support history-dependent coarse coverage. This run does not negate that result. It answers a different question: for the tested quotas and fixed scorers, restricting additions to predicted prefixes has no demonstrated rare-item advantage over simply adding full-TT candidates. The improvement over exact GenRec alone can therefore be explained without requiring prefix restriction. This is not proof that prefixes never help under other budgets or ranking functions.

Even full-TT slots5 yields only 63 tail/11 unseen hits, compared with standalone TT's 248/47. The hybrid preserves GenRec's head advantage at the cost of retaining only a small fraction of TT's rare-item capability. Top-10/20 remain unchanged by construction; all new hits are at positions 41–50 or 46–50. NDCG@50 follows the same trade-off: exact 2.3308%, full-TT slots5 2.3379%, full-TT slots10 2.3259%.

## Decision

Pause further prefix-hybrid tuning. Keep full-TT slots5 as the simple candidate if preserving aggregate/head quality is the priority; slots10 is a separate, explicitly stronger mid/tail trade-off. Neither should be described as a validated cold-start solution. Avoid selecting a quota solely because it wins a small observed validation difference.

Finish the already-running original-versus-permuted-SID control. That directly tests learned assignment utility and is logically separate from whether prefix retrieval improves this ensemble. Document both the positive coarse-coverage evidence and the negative incremental hybrid evidence. Any final comparison should use a fixed quota/objective and acknowledge earlier test-set exploration.

Runtime recorded for the evaluation was 21.85 minutes, including 113.21s real GenRec, 9.03s TT encoding/full scoring, and 1188.30s ranking/controls. These are diagnostic components, not a deployment latency comparison; prefix methods reused full-catalog TT scores.
