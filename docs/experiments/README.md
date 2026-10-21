# Research notebook

This is the project's research memory: what we asked, what we tried, what the
saved results show, and why the next question changed. Reconstructed on
**2026-09-19** from local artifacts, current source code, and the project discussion.
It is a working record, not a claim that the research is finished.

**Latest review (2026-10-17):** [project assessment and completion plan](reviews/project_review_20261017.md)
incorporates the 10,000-user prefix controls, model-comparison runs, research
papers, remaining validity gaps, and a bounded plan to finish the project.

Step 2 is implemented in the [prefix retrieval/ranking protocol](protocols/prefix_retrieval_ranking.md):
multi-seed popularity controls, shuffled histories, frozen learned Two-Tower
reranking/full-catalog scoring, and 5/10-slot hybrids. The
[full-validation results](results/prefix_retrieval_full_valid_20261017.md) now cover all
80,471 users: positive history-dependent prefix coverage, but no rare-item
advantage over full-catalog Content Two-Tower; quota hybrids show trade-offs.
The [full-TT hybrid control results](results/full_tt_control_20261018.md) now include
per-user paired checks: prefix restrictions do not improve observed tail/unseen
hits over the same full-TT quota hybrid; the five-slot full-TT hybrid's small
aggregate gain over exact GenRec is inconclusive.
An [offline analysis of prefix conditions](results/prefix_conditions_20261018.md) separates
rescued and excluded full-TT hits. Rare targets in prefixes with stronger
other-item training support show positive subsets, but exclusions and weak
within-pool ranking outweigh these gains overall; subgroup findings are exploratory.

The [semantic assignment control](protocols/sid_permutation_control.md) prepares matched
original/permuted Two-Tower SIDs and trains GenRec C with identical initialization
and epoch budgets. Its [completed validation results](results/sid_permutation_results_20261018.md)
show R@50 dropping from 6.607% to 3.453% after permutation, with zero mid/tail/unseen
top-50 hits in the permuted run. This supports learned assignment utility under
the matched budget, not superiority over Content Two-Tower.

## Reading guide

1. [Baseline development](results/baseline_development.md): the Recall@20 ≥ 0.10 goal,
   SASRec evaluation NaNs, split-order repair, all-prefix training, hyperparameters,
   GPU throughput, and the complete baseline Recall@20 ledger (B01–B09).
2. [Research log](research_log.md): hypotheses, assumptions, experiments,
   interpretations, decisions, corrections to earlier explanations, and open work.
3. [Results](results/overview.md): numerical tables for the main SID experiments, prefix
   survival, path likelihood, decoding interventions, and separate benchmarks.
4. [Earlier and archived experiments](archive.md): baseline hyperparameter runs,
   fine-tuning, retrieval augmentation, exact-ranking checks, and RecBole results.
5. [Evidence manifest](../../artifacts/main/derived/evidence_manifest.json): source artifact paths and SHA-256
   hashes at the time of this review. [Derived statistics](../../artifacts/main/derived/derived_statistics.json)
   records the small calculations made from existing arrays and diagnostic rows.
   [Baseline evidence](../../artifacts/main/derived/baseline_evidence.json) separately retains all 17 baseline
   histories' configurations, selected epochs, final metrics, and source hashes.
6. [Experiment template](protocols/experiment_template.md): copyable structure for a
   falsifiable prediction, fixed comparison, provenance, results, and decision.

The existing [SID ablation guide](protocols/sid_ablation.md) and
[path-likelihood/decoding guide](protocols/sid_path_likelihood.md) remain the runnable
protocols. The root [README](../../README.md) explains setup and pipelines.

The [main model-comparison pipeline](../../benchmarks/main-comparison/README.md)
adds ItemKNN, GRU4Rec, LightGCN, and an adapted TIGER baseline. Full main-test
outputs now exist for all four; TIGER's saved history currently contains only two
epochs and remains preliminary. See the [comparison](../../artifacts/benchmarks/main-comparison/reports/comparison.md)
and [latest review](reviews/project_review_20261017.md). Keep these runs separate from
the historical tables below.

## Questions and current answers

The earlier baseline questions and correctness fixes are indexed as **B01–B09**
in [baseline development](results/baseline_development.md#objective-and-status). The H IDs
below cover the subsequent capacity, SID, and generative-retrieval investigation.

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

## Evidence labels

- **Measured:** a saved metric or a calculation from an existing saved artifact.
- **Recorded:** reported in the discussion, without the original output recovered.
- **Implemented:** code exists; this alone does not establish a completed experiment.
- **Proposed:** an idea without an attributable completed result.
- **Interpretation:** an explanation consistent with observations, not a measured fact.

Most raw artifacts and checkpoints are ignored by Git. The Markdown tables retain
the important results in versionable documentation; raw-artifact links work only
in a workspace containing those files. Hashes identify the files reviewed, not
necessarily the checkpoint currently stored under a reused filename.

## Adding a future experiment

Start with the [experiment template](protocols/experiment_template.md).

Record the hypothesis and what would count against it **before** running. Add:
dataset/split and hashes; checkpoint/SID identity; exact command and configuration;
seed and user subset; candidate catalog, beam size and score normalization;
selection metric; baseline and treatment results by group; interpretation;
limitations; and the resulting decision. Use a new artifact name for each run.

Keep validation selection separate from test reporting. Preserve unsuccessful,
interrupted, and smoke runs, but label them so they cannot become accidental
baselines. Update the hypothesis table when evidence changes the explanation.
