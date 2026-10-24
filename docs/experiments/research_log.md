# Research log: semantic IDs, popularity, and generative retrieval

[Notebook index](README.md) · [Numerical results](results/overview.md) · [Archived work](archive.md)

Experiments are ordered by question, not assigned guessed dates. Some older results survive only as filenames. The underlying evidence was reviewed on 2026-09-19; this log did not require new training or evaluation.

## 1. Objective and assumptions

The target is a marketplace catalog where metadata is available even for products without training interactions. GenRec predicts semantic IDs from browsing history instead of keeping a separate learned embedding for every item.

The pipeline is `metadata → BGE → tokenizer → SID → GenRec`. Encoding an unseen item is possible; getting that item into top-k recommendations is a different problem.

### Evaluation contract

- The main data combines Electronics, Cell Phones and Accessories, and Office
  Products. The current snapshot has 80,471 users and 134,288 products.
- Each user's last two events are validation and test. Every earlier next-item
  prefix is a training example: 574,699 targets, `sequence[1:-2]`. The first
  training item is history-only. Test histories include the validation event.
- This is a per-user chronological split, not a common calendar cutoff. Filtering
  and catalog construction use the retained dataset before splitting.
- Tokenizers see metadata for the full retained catalog. The Two-Tower teacher
  learns from training interactions and supplies content-based vectors for all
  catalog items. “Unseen” means zero **training interactions**, not items/metadata that
  first appeared after model training. Dynamic insertion has not been evaluated.
- Popularity groups use all training interactions: head >20, mid 6–20, tail 1–5,
  unseen 0. These are absolute thresholds, not equal-size quantiles. A warm item
  can still have zero occurrences as a GenRec training **target**.
- GenRec uses catalog-constrained beam search and filters the complete seen
  history afterward. It is full-catalog candidate generation without sampled
  evaluation negatives, but not exhaustive exact ranking. Filtering may leave
  fewer than 50 recommendations. The main pipeline and benchmark pipelines have
  different duplicate/metadata filtering; see their READMEs.
- Recall@K has one held-out target per user, so it is also the hit rate. A zero tail R@10
  means no hits in that run, not that tail item is impossible to generate.

### Comparability rules

Comparisons need the same split, users, catalog, IDs, checkpoint, beam width, filtering, and score normalization. In particular, the Industrial result with 50,984 targets is not a baseline for the 80,471-user main dataset. Smoke, subset, validation, and test results must stay distinct.

Most results use one randomization seed and lack confidence intervals. Test data informed some early hypotheses, so treat these findings as exploratory; select future models on validation and reserve a fixed final test.

## 2. Baselines, implementation, and hyperparameters — H01–H03

The earlier [baseline development record](results/baseline_development.md) documents
the reproduced evaluation NaNs, timestamp-tie repair, training coverage changes,
GPU observations, and the original Recall@20 ≥ 0.10 target. It also distinguishes
historical procedures from current source and lists every retained baseline run.

**Question.** Is low GenRec quality an implementation problem, weak item content,
insufficient capacity, or the SID representation itself?

SASRec provides an ID-based sequential baseline. Content Two-Tower learns an item
network on top of BGE vectors plus a history encoder. Both current datasets use
64 sampled negatives per training prefix; GenRec instead learns per-position
SID-token cross-entropy. Training negatives are not the evaluation catalog.

Saved baseline histories include dimension, layer-count, history-length, and
batch-size variations. Larger dimensions did not consistently help. The full
tables and runs are in [the archive](archive.md#baseline-hyperparameter-runs).
In particular, a run with a lower validation selection score is not automatically
the teacher used later: the exported Two-Tower metadata identifies the actual
teacher as the 128-dimensional, three-layer, epoch-35 checkpoint.

**Text encoder hypothesis (considered).** A stronger initial embedding model or more
retained item text might improve semantic coverage. The current setup records
BGE-small, 384 **dimensions**, and a 512-**token** limit. Text also
passes through preprocessing truncation. This direction is neither validated nor
ruled out. It would require new embeddings, tokenizer training, SIDs, and GenRec.

**Embedding variants (implemented).** GenRec A learns semantic embeddings from
scratch; B initializes from RQ; C adds semantic input/output weight tying; D uses
a frozen RQ base plus regularized trainable deltas and tying. Most subsequent
experiments fix **GenRec C**. The separate SID ablation A/B/C labels do not select
these architecture variants. Older unlabelled GenRec histories cannot reliably
be mapped back to an A/B/C/D leaderboard.

### Decoder input

Current decoder inputs are `[BOS, c1, c2, c3]` for targets `[c1,c2,c3,c4]`, with
the previous token embedded using its own level's codebook and a causal mask.
Current evaluation rebuilds nonpersistent item SIDs from the matching file and
restores semantic-output configuration. Numeric code 42 from two tokenizers is
not the same vector: source-specific SID/RQ/GenRec files must stay paired.

## 3. SVD behavior-aware tokenization — H04

**Hypothesis.** Preserve semantic reconstruction while adding collaborative
structure to the quantized representation.

**Implementation.** Build a training-only user–item matrix, row-normalize it,
factorize its transpose with 64-dimensional truncated SVD, center warm-item
vectors, and normalize them. Fine-tune the content RQ-VAE with an auxiliary loss
on warm items. The current `BehaviorAwareRQVAE.forward` matches off-diagonal
pairwise cosine similarities of the **final quantized sum** to those of the SVD
teacher. A separate `forward1` method contains projected-vector cosine prediction;
it is not the active `forward` and should not be described as the same objective.

**What was tried.** Exported files exist for weights 0.001, 0.005, 0.01, and 0.05.
The locality comparison script explicitly compares 0.01 and 0.05. Recomputed
three-code churn against the saved content IDs is 69.16%, 79.26%, 88.33%, and
95.81%, respectively; [results](results/overview.md#svd-sid-churn) retain all four rows.

**Recorded result.** The discussion reports that weight 0.05 changed roughly 96%
of semantic SIDs without improving behavioral locality. The original printed
locality values were not recovered. Churn is independently confirmed from the
arrays; the locality conclusion remains discussion evidence, not a reconstructed
numerical result.

**Interpretation and decision.** Moving many assignments is not equivalent to
making useful prefixes. A loss on the full quantized sum need not make early hard
codes share behavioral neighborhoods. Pause this branch and try a teacher whose
geometry is learned directly for next-item recommendation. This does not prove
that all relational losses or SVD teachers are ineffective.

The locality utility samples a non-singleton prefix bucket uniformly and then
two items within it; it is not a uniformly sampled pair over all matching items.
It measures embedding geometry, not the probability that GenRec predicts a prefix.

## 4. Two-Tower representations as tokenizer inputs — H05

**Hypothesis.** BGE vectors transformed by a trained next-item item tower should
be easier to quantize into recommendation-relevant SIDs.

```text
metadata → BGE → trained content item tower → 128-d vector → RQ-VAE → SID
```

Unlike an ID-only collaborative lookup, the item network can process metadata
for an interaction-unseen product. That preserves an inductive representation
path, although evaluation still uses a known retained catalog.

**Experiment.** Export frozen Two-Tower vectors, train a separate three-level
RQ-VAE to reconstruct them, append a collision suffix, and train GenRec C with
the matching RQ initialization. Compare with the BGE/content SID baseline.

**Measured result.** Main test R@10 is 2.1809% for content and 2.3822% for Two-Tower;
R@50 is 5.3075% and 5.6691%. Most hits remain head items. Two-Tower has zero tail
and unseen top-10 hits in the reference prefix-survival run; content has four and
two. Better aggregate quality did not establish a solution for rare products.

The separate `genrec_twotower_C_test_metrics.json` has slightly different results
from the later prefix-survival file. Both remain listed in the archive; use the
prefix-survival file as the reference for the later SID/decoding tables rather
than silently merging them.

## 5. Prefix survival and the collision-token hypothesis — H06, H14

**Diagnostic.** Observe whether the correct target prefix remains after actual
beam pruning at each position. Do not inject the target into the beam. Rates use
all targets in the group, before seen filtering. Full-SID survival and filtered
item top-10 recall answer different questions.

**Initial Industrial observation.** Head survival was 100 → 49.49 → 35.17 →
29.97%; tail was 100 → 21.76 → 6.18 → 0.65%. Conditional on surviving position 3,
85.23% of head targets survived position 4, versus 10.53% of tail targets.
This made the arbitrary collision suffix look like a major bottleneck.

**Sanity check.** The saved Industrial SID array has **66 root children**. Beam
100 can keep all of them, explaining 100% first-position survival without
requiring perfect ranking. Main Two-Tower has 256 root children and 84.28%
first-position survival. Four-RQ B/C have only 73/80 root children and therefore
also get 100% at position 1 with beam 100.

**Correction to the early interpretation.** Survival loss at position 4 does not
isolate the quality of the fourth token. Beam search globally compares cumulative
path scores from different parents; branching and competition change at each
step. A target can be pruned even when its own final transition is fairly likely.
Similarly, 100% c1 survival does not prove c1 scores are good: those scores still
contribute to later cumulative ranking.

**Beam width evidence.** Industrial beam 100 and beam 500 have identical R@10
(3.6972%); R@50 changes only from 9.6187% to 9.6364%. An archived 16-user exact
ranking diagnostic found 99% head items in exact top-50 lists. Neither is a
population-wide proof, but both argue against explaining the gap solely by
insufficient beam width. See [archived diagnostics](archive.md).

## 6. Four learned codes and uniqueness-aware export — H07

**Prediction before the run.** Replacing `[c1,c2,c3,collision]` with four semantic
codes might preserve meaningful final refinements and raise tail full-SID survival
without extending the autoregressive sequence.

**Uniqueness.** Four RQ codes can still collide. During export, fix the first three
codes and match items within each bucket to distinct fourth-level codewords:

$$
\min_{a_i}\sum_i\|r_i-e_{a_i}\|^2,
\qquad a_i\ne a_j\text{ for distinct items in the same bucket}.
$$

Here `r_i = encoder(x_i) - e(c1) - e(c2) - e(c3)`. Hungarian assignment minimizes
total squared distance within the bucket. Each bucket must contain at most the
fourth codebook's 256 entries. Training remains ordinary RQ; uniqueness is an
export-time constraint, not a differentiable training objective.

**Experiment A.** Keep Two-Tower input and reconstruction supervision, train four
RQ levels, export unique four-token SIDs, then train GenRec C. All four positions
are initialized and tied. This also changes tokenizer capacity, the learned early
assignments, the final vocabulary, and final-position tying relative to the old
collision model; it is not a surgical replacement of only c4 with frozen c1–c3.

**Measured result.** Raw collision rate 3.15% became zero; the largest three-code
bucket had 49 items; 4,342 fourth assignments changed. But R@10 fell from 2.3822%
to 2.2182%. Tail full survival fell from 0.1956% to 0.0682%, and unseen full
survival became zero. The hoped-for gain did not appear.

**Decision.** Keep the 4 learned codes with uniqueness export as an experiment, but stick with three learned codes and a fourth collision suffix.

## 7. Semantic reconstruction plus behavioral supervision — H08

**Hypothesis.** Keep BGE semantics while teaching the partial quantized sums to
predict a frozen Two-Tower representation. This is different from reconstructing
Two-Tower vectors directly.

- **B / `rq4_b`:** BGE input and reconstruction; behavioral loss on the final sum.
- **C / `rq4_c`:** same input, architecture, shared projection, and teacher;
  behavioral loss averaged over all four partial sums.

The implemented loss is reconstruction MSE + VQ loss + 0.05 × mean projected
cosine loss. C averages depths equally; it does **not** use the earlier suggested
weights `1, 0.5, 0.25`, and it does **not** implement pairwise relational matching
at every depth. The teacher is Two-Tower, not SASRec. Straight-through gradients
train the encoder/projection; codebooks receive their VQ updates.

**Measured result.** B/C R@10 is 2.2107%/2.2604%, both below the 2.3822% Two-Tower
reference. Both have zero tail and unseen top-10 hits. B substantially improves
tail prefix-2 survival (29.65% versus 11.37%) but that becomes just 0.1097%
full-SID survival. Better early survival did not carry through to final retrieval.

**Interpretation.** The hierarchy changes, but this loss/weight/selection setup
does not solve the observed bias. B/C's small number of root codes explains their
perfect c1 survival. Higher early survival by itself is not sufficient evidence
of stronger personalization. Tokenizers were selected by reconstruction loss,
not downstream recall, and weight 0.05 was not a validated optimum.

**Decision.** Pause new SID variants and study the recommendation model's path
scores. This is a prioritization decision, not proof that SID construction never
matters or that another teacher/objective could not help.

## 8. Teacher-forced path probabilities — H09

**Hypothesis.** Autoregressive likelihood favors frequent training paths even
after conditioning on history. Investigate each true transition, including paths
that beam search would prune.

**Protocol.** Use the same main Two-Tower GenRec C checkpoint and all test targets.
Record `log P(c_t | history, true previous tokens)` and cumulative prefix log
probabilities. Count actual training-target occurrences, not held-out events or
all training history events. Compare real histories with a fixed shuffled-user
permutation. The diagnostic uses full-vocabulary softmax without trie normalization.

**Measured result.** Mean head token log probabilities are approximately
`[-4.478, -2.921, -1.050, -0.224]`; tail values are
`[-4.667, -5.133, -5.312, -0.450]`. Positions 2 and 3 account for about **94%**
of the difference in mean head versus tail full-path log probability. The fourth
position's average gap is comparatively small. This weakens the claim that the
collision token alone explains the main dataset's struggle.

About **88%** of the total mean real-minus-shuffled log-probability benefit occurs
at c1. The conditional empirical log prior and model token log probability have
Spearman correlation **0.927 at c2**. Marginal token counts, complete-prefix
counts, and conditional priors are different quantities; the tables distinguish
them rather than treating every correlation as the same measurement.

**Interpretation.** Later scores are strongly associated with training frequency,
and history changes their average likelihood much less than it changes c1.
That is consistent with strong learned path priors. It does not prove popularity
is the sole cause: legitimate demand, history, and frequency are correlated.
Teacher forcing supplies a highly informative true prefix at later positions;
small shuffled-history effects there do not prove that the model ignores users.
This is a likelihood diagnostic, not a formal calibration test.

## 9. Conditional-prior correction — H10–H11

**Intervention.** During beam expansion, score a legal child with

$$
\log P_\theta(c_t\mid h,c_{<t})
-\alpha_t\log\frac{N(c_{\le t})+\beta}
{N(c_{<t})+\beta\,K(c_{<t})},\qquad \beta=0.5.
$$

`K` is the number of catalog-valid children. Counts come from the 574,699 training
targets. Unseen legal paths receive finite smoothed priors. Correction happens
before pruning, not just when reranking final beams; corrected scores are not
normalized probabilities. Alpha=0 is the uncorrected baseline.

**All-level sweep.** Validation alpha values were 0, 0.05, 0.1, 0.2, 0.3, and 0.5.
Every nonzero value reduced aggregate R@10 and NDCG@10 relative to alpha zero.
At 0.5, validation R@10 dropped from 2.8246% to 2.1548%; non-head top-10 hits rose
from 30 to 69. The saved test run at 0.5 similarly trades head performance for
more rare hits: R@10 2.3822% → 1.8317%, non-head hits 17 → 53.

That test alpha is a trade-off probe, **not** the winner under aggregate validation
R@10 or NDCG@10. Under those criteria, this sweep selects alpha zero.

**c2-only sweep.** Set `[alpha_1,alpha_2,alpha_3,alpha_4] = [0,alpha,0,0]`.
At 0.5, validation R@10 is 2.2766%, with 42 non-head hits. It loses less overall
recall than all-level correction, but also recovers fewer non-head hits. No
nonzero c2-only alpha beats the baseline's aggregate validation R@10/NDCG@10.

**Decision.** Neither sweep establishes a better aggregate recommender. Correction
can expose a head/rare-item trade-off, but does not “prove priors drown all
personalization,” and a negative sweep does not prove the absence of rare-item
signal. Smoothing, per-level choices, and the model distribution still matter.

## 10. Full vocabulary versus valid-child normalization — H12

**Observation motivating the experiment.** Even when a c1,c2 prefix has exactly
one valid c3 child, the full-vocabulary softmax can assign little mass to it.
Among such target rows, mean probability on that sole valid child is 83.72% for
head, 17.22% for mid, 2.99% for tail, and 1.44% for unseen. These are means of
probabilities, not exponentiated means of log probabilities. Sample sizes and
source hashes are retained in the results/derived statistics.

**Mechanism.** Default decoding computes softmax over the full level vocabulary
and then masks illegal children. The alternative masks logits before softmax:

$$
\log P_{\rm valid}(c\mid h,p)
=\log P_\theta(c\mid h,p)-\log\sum_{v\in V(p)}P_\theta(v\mid h,p).
$$

This preserves the order of children under the same parent but changes scores
between different parents. A singleton child contributes log probability zero.
The correction depends on the current history and prefix; it is not the same as
subtracting the empirical training prior. The tested mode uses alpha zero and
does not retrain the model.

**Measured result.** Validation R@10 changes 2.8246% → 2.7936%; NDCG@10 changes
1.5078% → 1.4961%; non-head top-10 hits rise 30 → 44. Test R@10 is exactly equal
at 2.3822% (1,917 hits), with non-head hits increasing 17 → 39 and head hits
decreasing 1,900 → 1,878. Test NDCG@10 and R@50 decrease slightly.

**Interpretation.** This is a relatively mild redistribution with some rare-item
recovery, not a drastic improvement or a complete fix. For the main tokenizer,
all 256 c1 values are valid, so normalization leaves first-position probabilities
and survival unchanged. Renormalizing also discards information in the model's
total legal mass; a prefix can gain even if all its legal children had low raw
probability. It is a decoding ablation, not automatically a correctness repair.

**Decision.** Preserve the option and its separate output files. Full-vocabulary
decoding remains the stronger aggregate validation result among these two modes.
The earlier archived `normalize_valid` fine-tuning trials changed training too;
do not treat them as repeats of this same-checkpoint decoding-only experiment.

## 11. Other datasets and parallel avenues

MovieLens 1M, Beauty and Personal Care, and Industrial and Scientific have separate
pipelines and dataset-specific RQ-VAE training. Reusing main-dataset codebooks for
a different dataset was explicitly rejected; a numeric SID has meaning only in
its matching tokenizer.

MovieLens has a saved full GenRec A test run: R@10 28.1623%, on 6,040 users. Its
head group contains 5,962 targets and unseen only two under the fixed Amazon-style
thresholds, so this is not evidence of solved marketplace cold start. Industrial
has full C evaluations and prefix diagnostics. Beauty has a saved 10-user smoke
evaluation; that is not a benchmark-quality result. See [benchmark tables](results/overview.md#separate-benchmarks).

The archive also contains learned fine-tuning, recent-sampling, collaborative and
directional retrieval trials, and a RecBole SASRec comparison. They are retained
in [their own record](archive.md) because their checkpoints, populations, and
available implementation evidence differ from the current SID ablation.

## 12. What remains open

1. **Improve later history conditioning.** The diagnostic motivates this question,
   but a particular architecture or objective has not yet been validated.
2. **Align checkpoint selection with retrieval.** Current main GenRec uses token
   loss; SID B/C tokenizers use reconstruction loss. Archived fine-tuning used
   validation NDCG. Isolate selection from training changes in a future comparison.
3. **Test stronger content encoders and text coverage.** Measure actual truncation
   first; dimension 384 is not a token budget. No completed controlled run is logged.
4. **Define the optimization goal.** Maximizing aggregate recall and improving rare
   groups can conflict. A rare-item objective or constraint should be chosen before
   selecting correction weights, not invented after inspecting test results.
5. **Recover missing provenance.** Older locality outputs, unlabelled GenRec variant
   runs, and archived runner code remain gaps. Do not replace them with guesses.
6. **Validate robustness and deployment assumptions.** Repeat seeds, freeze a final
   evaluation protocol, inspect seen-target eligibility, and test SID stability for
   actual catalog additions before making dynamic-catalog claims.

SID choices change routing, but four-code uniqueness and the tested behavioral losses did not close the popularity gap. Path-score diagnostics exposed another issue; decoding changes recovered a few rare hits at the cost of aggregate quality. New experiments should isolate one explanation at a time.
