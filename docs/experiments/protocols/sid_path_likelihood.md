# Main dataset: SID likelihood versus training frequency

Commands for path-likelihood diagnostics, training-frequency priors, and decoding ablations. See the [research log](../research_log.md#8-teacher-forced-path-probabilities--h09) for motivation and [results](../results/overview.md#path-likelihood-and-history-sensitivity) for outcomes.

Teacher forcing scores every held-out SID, including paths beam search would prune. It does not change training or decoding and does not run beam search.

```bash
python -m scripts.evaluate_genrec --sid-source twotower --split test \
  --batch-size 64 --diagnose-paths --shuffle-histories
```

Use `--sid-source content`, `rq4_a`, `rq4_b`, or `rq4_c` for other checkpoints. Add `--limit 100` for a smoke run; frequency counts still use all training users. `--shuffle-histories` scores a second, shuffled history for each user (at least two users required). `--beam-size` is ignored here.

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

Logs are natural. Token probabilities use the model's full vocabulary, before trie masking. No seen-item filtering applies; teacher forcing supplies previous true tokens under the decoder's causal mask.

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

Path-frequency correlation alone does not show that popularity causes retrieval errors. Compare within groups and against shuffled-history scores; this is not a calibration study.

A conditional-prior correction has to run during beam expansion. Reranking completed beams cannot recover pruned candidates. Low-frequency and zero-count paths may receive large boosts.

## Conditional-prior decoding sweep

Run the conservative alpha sweep on validation data with the same checkpoint:

```bash
python -m scripts.evaluate_genrec --sid-source twotower --split valid \
  --beam-size 100 --batch-size 64 --diagnose-prefixes \
  --prior-alpha-sweep 0 0.05 0.1 0.2 0.3 0.5
```

`--prior-alpha-sweep` uses the shown grid, including alpha 0. Add `--limit 100` for a smoke run. Each alpha gets a separate beam evaluation, with priors built once; training is unchanged.

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

An improvement would support this decoding rule, not prove a causal theory of personalization. A flat result would not rule out other priors, per-level weights, or normalization.

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

Trie normalization changes the decoding objective, not training. It removes probability mass assigned to impossible continuations, but may also promote weak prefixes. Compare group-level Recall/NDCG and survival on validation before trying test. `--diagnose-paths` always uses raw model probabilities.
