# When prefix restriction helps rare targets — 2026-10-18

Sources: [full-TT control analysis](full_tt_control_20261018.md), [per-user derived statistics](../../../artifacts/main/derived/prefix_conditions_20261018_statistics.json). This is an offline analysis of the existing 80,471-user validation run; no inference or training was performed.

## Scope

Compare pure Two-Tower ranking of all catalog items with the SAME scores restricted to expanded GenRec prefixes. These are standalone top-50 lists, not the 5/10-slot quota hybrids. Input hashes and ordered users/targets were verified against current SID and sequence files.

For a covered target, removing candidates cannot worsen its rank with identical scores and tie-breaking. Rank monotonicity holds for all observed hits shared by full/prefix TT. Consequently, a full-TT hit lost by prefix TT must have been excluded from the prefix candidate pool. This inference would not apply if scores were mixed with prefix probabilities or the ranker changed.

## Exact retrieval/ranking decomposition

| Depth | Group | Covered targets | Full-TT hits | Retained hits | Rescued from outside full-TT top-50 | Lost full-TT hits | Prefix-TT hits | Covered but missed by ranker |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | all | 17550 | 4139 | 3732 | 499 | 407 | 4231 | 13319 |
| 2 | head | 9900 | 3531 | 3330 | 354 | 201 | 3684 | 6216 |
| 2 | mid | 3381 | 313 | 235 | 75 | 78 | 310 | 3071 |
| 2 | tail | 3785 | 248 | 147 | 63 | 101 | 210 | 3575 |
| 2 | unseen | 484 | 47 | 20 | 7 | 27 | 27 | 457 |
| 3 | all | 8571 | 4139 | 3304 | 1442 | 835 | 4746 | 3825 |
| 3 | head | 7387 | 3531 | 3147 | 1180 | 384 | 4327 | 3060 |
| 3 | mid | 763 | 313 | 109 | 179 | 204 | 288 | 475 |
| 3 | tail | 380 | 248 | 44 | 76 | 204 | 120 | 260 |
| 3 | unseen | 41 | 47 | 4 | 7 | 43 | 11 | 30 |

For depth-2 tail: 248 full-TT hits − 101 excluded hits + 63 rescued hits = 210 prefix-TT hits. For unseen: 47 − 27 + 7 = 27. Prefix restriction genuinely rescues some rare targets, but exclusions cost more. At depth 3 the corresponding rescue/loss counts are 76/204 for tail and 7/43 for unseen.

Depth-2 ranking converts only 210/3785 = 5.55% of covered tail targets and 27/484 = 5.58% of covered unseen targets to hits. Thus both candidate selection and ranking are bottlenecks. Depth-3 conditional recall is higher (tail 120/380 = 31.58%; unseen 11/41 = 26.83%), but much less coverage makes its total recall worse. The changed candidate populations prevent treating those conditional rates as isolated improvements to the scorer.

## Observed condition: training support from other items in the prefix

For each target, count actual training next-item events for OTHER items sharing its prefix, using seq[1:-2] over all users. Exclude the target item itself. This is a count of training events, not distinct neighbors, and contains no validation/test interactions.

Depth 2, tail targets:

| Other-item training events | n | Full-TT hits | Prefix-TT hits | Rescued | Lost | R@50 full → prefix |
|---|---:|---:|---:|---:|---:|---|
| 0 | 3830 | 18 | 0 | 0 | 18 | 0.470% → 0.000% |
| 1–20 | 16201 | 82 | 35 | 14 | 61 | 0.506% → 0.216% |
| 21–100 | 8148 | 67 | 70 | 24 | 21 | 0.822% → 0.859% |
| 101+ | 4100 | 81 | 105 | 25 | 1 | 1.976% → 2.561% |

Tail targets with >100 other-item events benefit observationally: 105 versus 81 hits. They account for only 4,100/32,279 = 12.7% of tail targets. Those with 1–20 events account for 16,201 targets and lose 47 net hits. A useful subset does not guarantee a useful aggregate method.

Mid shows the same direction at depth 2: >100 other-item events gives 168 prefix hits versus 135 full TT; 1–20 gives 33 versus 66. Unseen has 9 versus 5 hits in the >100 subset (460 targets), but 10 versus 29 in the 1–20 subset (2,365 targets). The high-support unseen difference is only four hits; do not treat it as robust confirmation.

At depth 3, 26,403/32,279 tail targets and 3,796/4,482 unseen targets have zero other-item training-target support in their prefix. Thus deep addresses often cannot benefit from interactions with another item in that exact prefix. This does not rule out transfer through shared individual tokens or earlier prefixes.

## Observed condition: catalog bucket size

Depth 2, tail targets. Size means the target's full catalog prefix bucket before seen filtering, not the entire candidate pool across all beam prefixes.

| Items in target bucket | n | Full-TT hits | Prefix-TT hits | Rescued | Lost |
|---|---:|---:|---:|---:|---:|
| 1 | 3499 | 17 | 0 | 0 | 17 |
| 2–10 | 20193 | 117 | 71 | 24 | 70 |
| 11–50 | 6730 | 87 | 103 | 30 | 14 |
| 51+ | 1857 | 27 | 36 | 9 | 0 |

Bigger target buckets correlate with better net retrieval effects here, plausibly because they have more supported paths. This is not evidence that increasing every bucket size will help: bucket size, prefix frequency, item/category distribution, and learnability are confounded. Expansion also increases ranking competition. The current NPZ does not contain per-user coverage flags or pool sizes, so coverage conditional on these strata cannot be reconstructed; the tables report final hits and inferred retrieval losses.

History-length strata 2–4, 5–9, and 10+ do not reverse the negative aggregate effect for tail or unseen at either depth. For mid, histories of 10+ show a positive net difference (depth 2: 71 versus 63 hits; depth 3: 75 versus 63), an exploratory association rather than a validated gating policy. All stratified counts are in the linked JSON.

## Interpretation and limits

The results support a bounded mechanism: useful prefix selection can remove competitors and rescue targets, particularly when other products in the same prefix have training support. However, weakly supported prefixes exclude many rare targets, and most covered rare targets still rank below top-50. This can explain why positive coarse coverage does not produce a rare-item advantage over full-catalog Content Two-Tower. It does not establish that support alone causes the benefit or that semantic transfer is its sole mechanism.

The semantic-assignment permutation control separately supports learned assignment utility for GenRec C. It cannot be used to attribute every stratified effect here to pure text semantics: the tokenizer is Two-Tower-derived, and the permutation changes both input and output correspondence.

These thresholds and strata were examined on the already-used validation set; results are descriptive, with no multiple-comparison-adjusted significance claim. Conditions are defined using the known target, which is unavailable during recommendation. An oracle that chooses a method based on the eventual target bucket would be invalid. A deployable decision must use history/candidate information and be validated independently.

## Bounded next step

Freeze these descriptive strata and check them on test using saved per-user outputs if the project proceeds to a final evaluation; acknowledge prior test exploration. Existing evaluator exports are sufficient for the hit rescue/loss analysis. No new tokenizer or training objective is justified solely by these subgroup observations. If a later study needs conditional candidate coverage or ranks beyond 50, save those per user in the next scheduled evaluation rather than inferring them from zeros in target_rank.
