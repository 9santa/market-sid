# Main dataset: SID likelihood versus training frequency

This page defines the diagnostics and commands. The [research log](../research_log.md#8-teacher-forced-path-probabilities--h09)
records why they were introduced and how the conclusions changed. The
[saved results](../results/overview.md#path-likelihood-and-history-sensitivity) include path
likelihood, prior sweeps, and the completed trie-normalization experiment.

Use the existing tokenizer and GenRec checkpoint. This diagnostic changes neither
training nor decoding scores, and runs no beam search. It scores every held-out
target with teacher forcing, including paths that normal beam search would prune.

```bash
python -m scripts.evaluate_genrec --sid-source twotower --split test \
  --batch-size 64 --diagnose-paths --shuffle-histories
```

Use `--sid-source content`, `rq4_a`, `rq4_b`, or `rq4_c` for the other checkpoints.
For a small run add `--limit 100`. The frequency counts still use the full training
population. The optional shuffled-history comparison doubles scoring work and
uses a fixed permutation with no user receiving their own history. It requires
at least two evaluation users. It is a diagnostic comparison, not an unconditional
model probability. `--beam-size` has no effect in this mode.

Outputs share the stem `artifacts/main/diagnostics/paths/genrec_twotower_C_test_path_diagnostics`:

- `.parquet`: one row per target with user/item IDs, group, SID tokens, each
  conditional token log probability, cumulative prefix log probabilities,
  marginal token counts, joint prefix counts, parent counts, and conditional
  training log priors. Includes shuffled scores when requested.
- `.json`: mean/median token log probabilities by all/head/mid/tail/unseen,
  prefix-frequency and token-frequency correlations, zero-count observations,
  and real-minus-shuffled mean token log probabilities. Prefix correlations are
  reported both per target and with equal weight per distinct observed prefix.
- `.png`: group-specific mean log probabilities in log-frequency bins: joint
  prefix count versus cumulative prefix log probability, and marginal token
  count at a level versus conditional token log probability.

All logarithms are natural. Probabilities use the full model vocabulary before
trie masking, matching production beam-search scores. No seen-item filtering is
applied. Teacher forcing supplies only preceding true tokens because the decoder
uses its causal mask.

## Count the training objective's actual targets

The dataset creates one target for each position in `sequence[1:-2]`. The first
interaction is history-only; the last two are validation/test. Count these target
occurrences, not every interaction and not just unique items. Marginal token
counts are separate for each SID position. Prefix counts use the complete prefix,
so token 42 under two different parents has two different conditional priors.

Popularity groups retain the existing evaluation definition from all training
interactions. Thus a history-only item can belong to a warm group while having
zero training-target count. Both notions are intentionally kept separate.

For a prefix with count N and parent count M, the diagnostic saves
`log((N + beta) / (M + beta * valid_children))`, with `beta=0.5` by default
(`--prior-smoothing`). Valid children come from the full catalog, including paths
with no training targets. This smoothing affects only the reported empirical
prior; it does not alter model log probabilities. Singleton catalog branches
have conditional log prior zero, even when the model assigns them low probability.

## Interpretation before correction

A frequency correlation is evidence of association, not proof that popularity
alone causes the error. Histories and legitimate next-item demand are correlated
with popularity too. Inspect within-group trends, equal-prefix-weighted results,
and how much real history improves scores relative to shuffled history. This
measures likelihood/frequency dependence; it is not a full calibration test.

A later correction could subtract per-level smoothed log priors, with alpha
selected on validation data. It must be applied during beam expansion to affect
early pruning; reranking final beams cannot recover missing candidates. Watch
zero-count paths carefully, since subtracting their small log priors can create
large boosts.

## Conditional-prior decoding sweep

Run the conservative alpha sweep on validation data with the same checkpoint:

```bash
python -m scripts.evaluate_genrec --sid-source twotower --split valid \
  --beam-size 100 --batch-size 64 --diagnose-prefixes \
  --prior-alpha-sweep 0 0.05 0.1 0.2 0.3 0.5
```

Bare `--prior-alpha-sweep` selects this same grid. Alpha zero is always included.
For a smoke run, add `--limit 100`; priors still use every training user. The
checkpoint and priors are loaded once, then each alpha runs its own beam search.
This requires six decoding evaluations, with no retraining.

At every expansion, the score increment is
`log P_model(token | history, prefix) - alpha * log P_train(token | prefix)`.
The correction uses all SID levels, before pruning, with the same smoothed prior
definition as above. It does not subtract marginal token frequency, does not
renormalize model logits over valid children, and does not rerank only the final
beams. Alpha zero bypasses the correction entirely. Illegal children remain
masked; zero-count legal paths have finite smoothed priors. Corrected scores are
ranking scores, not normalized log probabilities.

Outputs are `artifacts/main/diagnostics/decoding/genrec_twotower_C_valid_prior_sweep.json` and `.csv`.
JSON includes each alpha's full group metrics and optional survival rates; CSV
has one row per alpha/group. Completed alpha runs are saved to JSON as the sweep
progresses. Existing normal-evaluation files are preserved. `--output path.json`
can distinguish grids or smoothing settings.

Select alpha by a stated validation criterion and inspect head/mid/tail/unseen
trade-offs. Then evaluate the fixed choice on test; for example, **only if 0.2
was selected on validation**:

```bash
python -m scripts.evaluate_genrec --sid-source twotower --split test \
  --beam-size 100 --batch-size 64 --diagnose-prefixes --prior-alpha 0.2
```

This saves `artifacts/main/diagnostics/decoding/genrec_twotower_C_test_prior_alpha0.2.json` and `.csv`.
For either mode, `--prior-smoothing` defaults to 0.5. The alpha flags are mutually
exclusive and cannot be combined with teacher-forced `--diagnose-paths`.

Improvement would show that this score correction helps retrieval, not by itself
prove a causal explanation of personalization. No improvement on this grid
would not establish that the model contains no useful rare-item signal; smoothing,
the shared alpha across levels, and invalid-continuation penalties also matter.

## Isolate correction at c2

Use the same grid, applying it only to the second SID position:

```bash
python -m scripts.evaluate_genrec --sid-source twotower --split valid \
  --beam-size 100 --batch-size 64 --diagnose-prefixes \
  --prior-levels 2 --prior-alpha-sweep 0 0.05 0.1 0.2 0.3 0.5
```

This uses per-level alphas `[0, alpha, 0, 0]`. The model's score increments at
c1/c3/c4 remain uncorrected; later beams can still change because they inherit
the corrected c2 score. Alpha zero is the unchanged baseline.

Outputs are `artifacts/main/diagnostics/decoding/genrec_twotower_C_valid_prior_sweep_levels2.json` and
`.csv`, separate from the all-level sweep. JSON records the selected positions
and each run's alpha vector; CSV includes `alpha_c1` through `alpha_c4`.

`--prior-levels` uses one-based positions and also accepts multiple levels, e.g.
`--prior-levels 2 3`. Omitting it preserves correction at all levels.
After selecting a validation alpha, use the same `--prior-levels 2` with
`--prior-alpha VALUE --split test`. Add `--limit 100` for a smoke evaluation.

## Normalize over trie-valid children

This separate experiment uses alpha zero and the existing checkpoint. Mask
invalid logits **before** softmax, so valid children sum to probability one at
each prefix. A singleton branch adds zero to the log score. Ordering within a
prefix is preserved, but competition between different prefixes changes.

```bash
python -m scripts.evaluate_genrec --sid-source twotower --split valid \
  --beam-size 100 --batch-size 64 --diagnose-prefixes --trie-normalize
```

Results go to
`artifacts/main/evaluation/genrec_twotower_C_valid_prefix_survival_trie_normalized_metrics.json`.
The file records `trie_normalize: true` and `prior_alpha: 0.0`. Compare with the
alpha-zero validation metrics in the earlier prior sweep. No prior flags are
needed. This changes only decoding normalization; it does not retrain the model
or change its training loss.

For a fresh full-vocabulary baseline, run the same command without
`--trie-normalize`. Add `--limit 100` to both commands for a smoke comparison.
Different normalization modes have separate default output filenames.

Trie normalization changes the decoding distribution; it is an ablation, not an
automatic correctness fix. It removes penalties for mass placed on impossible
continuations, but can also promote a prefix whose legal continuations all have
low model probability. Compare Recall/NDCG and survival by group on validation
before evaluating the selected mode on test. Teacher-forced `--diagnose-paths`
continues to measure raw model probabilities and cannot use this flag.
