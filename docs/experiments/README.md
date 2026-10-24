# Research notebook

Start here for the experiment history and current findings. This index was assembled from code and saved results on 2026-09-19; newer validation controls are linked below.

The main result is mixed. GenRec prefixes respond to user history, and the learned SID assignment beats a matched permutation. But **rare-item recommendations have not beaten full-catalog Content Two-Tower**, and putting Two-Tower items behind a prefix restriction has not improved the tail/unseen hybrid result.

## Recent results

- [Full-validation prefix retrieval](results/prefix_retrieval_full_valid_20261017.md): 80,471 users, multi-seed popularity controls, shuffled-history control, learned Two-Tower ranking, and 5/10-slot hybrids.
- [Full-TT hybrid control](results/full_tt_control_20261018.md): restricting the same Two-Tower scores to prefixes did not increase tail/unseen hits over full-catalog additions.
- [When prefixes help](results/prefix_conditions_20261018.md): some rare targets are rescued by restricting candidate pools, but more are excluded or poorly ranked. The subgroup findings are exploratory.
- [SID permutation](results/sid_permutation_results_20261018.md): R@50 fell from 6.607% to 3.453% after reassigning whole SIDs within training-frequency bins. A single matched seed pair supports assignment utility, not superiority over other recommenders.
- [Project review](reviews/project_review_20261017.md): the outstanding validity checks and completion plan.

## Where to look

- [Baseline development](results/baseline_development.md) — SASRec/Two-Tower fixes, Recall@20 runs, and throughput.
- [Research log](research_log.md) — hypotheses, failures, changes of direction, and open questions.
- [Results overview](results/overview.md) — main SID comparisons, path probabilities, decoding, and other datasets.
- [Archive](archive.md) — older hyperparameter sweeps and exploratory runs.
- [Protocols](protocols/experiment_template.md) — reusable experiment template. See also [SID ablations](protocols/sid_ablation.md), [path likelihood](protocols/sid_path_likelihood.md), [prefix ranking](protocols/prefix_retrieval_ranking.md), and [SID permutation](protocols/sid_permutation_control.md).
- [Model comparisons](../../benchmarks/main-comparison/README.md) — ItemKNN, GRU4Rec, LightGCN, and adapted TIGER on the main split. The saved TIGER history has only two epochs.

The [evidence manifest](../../artifacts/main/derived/evidence_manifest.json), [derived statistics](../../artifacts/main/derived/derived_statistics.json), and [baseline ledger](../../artifacts/main/derived/baseline_evidence.json) point to the saved files and hashes behind these notes.

## Questions still being tested

| ID | Hypothesis or question | Current status |
|---|---|---|
| H01 | Bigger recommendation models or longer histories improve quality | Mixed small-run evidence; increasing dimension did not consistently help |
| H02 | Better text embeddings / longer text fix weak SIDs | Proposed; no controlled encoder comparison recovered |
| H03 | RQ initialization and semantic weight tying help GenRec | Implemented A/B/C/D; no fully attributable main-dataset A/B/C/D result table recovered |
| H04 | SVD behavioral supervision improves SID locality | High SID churn measured; no locality improvement reported in discussion; branch paused |
| H05 | Two-Tower representations make better recommendation SIDs | Higher aggregate recall than BGE in the saved main comparison, but not a long-tail solution |
| H06 | Arbitrary collision IDs are the dominant bottleneck | Strong initial suspicion; four-RQ results and path likelihood weakened it |
| H07 | Four semantic codes with unique assignment solve final-step losses | Uniqueness achieved; expected retrieval improvement not observed |
| H08 | Behavioral supervision at every RQ depth improves routing | Implemented with projected cosine loss; no decisive gain over final-depth supervision |
| H09 | Popularity-related path scores overwhelm personalization | Strong association and uneven history sensitivity measured; causality not established |
| H10 | Conditional-prior subtraction improves retrieval | More rare-item hits at large correction, worse aggregate validation metrics |
| H11 | Correcting only c2 avoids the trade-off | Less aggregate damage at alpha 0.5, but still below the uncorrected validation baseline |
| H12 | Normalizing over trie-valid children helps rare paths | Modest redistribution; same test R@10, slightly worse validation aggregate metrics |
| H13 | Fine-tuning / collaborative retrieval can help beyond SID changes | Positive archived subset results, with multiple changed settings and missing runner code |
| H14 | The problem is only approximate beam search | Small exact-ranking diagnostic and Industrial beam sweep argue against this explanation |
| H15 | GenRec prefixes use history to cover rare targets | Supported by full-validation shuffled-history and frequency/size controls; not isolated semantic causality |
| H16 | Prefix restriction adds value to a TT quota hybrid | No observed tail/unseen advantage over the same full-TT hybrid at 5/10 slots |
| H17 | Learned item-to-SID assignments help GenRec C | Large positive matched-budget effect versus popularity-stratified permutation; one seed pair, validation only |

## Reading results

*Measured* means the number comes from a saved output. *Recorded* means the original output could not be recovered. *Implemented* only says code exists; *proposed* has not been tested. Explanations are interpretations, not measurements.

Most raw outputs are Git-ignored, so some artifact links require a local research workspace. Hashes identify the files reviewed at the time, not necessarily a checkpoint later written to the same filename.

For new work, start with the [experiment template](protocols/experiment_template.md). Fix the split, population, candidates, model and SID hashes, seed, metric, and stopping rule before running. Select on validation, keep test separate, give each run a new filename, and record negative or interrupted runs alongside successes.
