# Repository and result layout

Code, documentation, model weights, and measurements have separate homes.
Commands are run from the repository root.

| Location | Contents |
|---|---|
| `docs/experiments/protocols/` | Experiment assumptions, procedures, and runnable commands |
| `docs/experiments/results/` | Human-written interpretation of saved results |
| `docs/experiments/reviews/` | Project assessments and completion plans |
| `docs/experiments/research_log.md` | Hypotheses, decisions, and research history |
| `artifacts/main/training/` | GenRec, SASRec, Two-Tower, and RQ training histories |
| `artifacts/main/evaluation/` | Main-dataset Recall/NDCG and prefix survival metrics |
| `artifacts/main/diagnostics/prefix_retrieval/` | Prefix retrieval, hybrid/control results, and per-user NPZ files |
| `artifacts/main/diagnostics/paths/` | Teacher-forced SID path likelihood diagnostics |
| `artifacts/main/diagnostics/decoding/` | Conditional-prior decoding sweeps |
| `artifacts/main/tokenizers/` | SID export statistics and locality logs |
| `artifacts/main/reports/` | Generated comparison tables |
| `artifacts/main/derived/` | Small evidence manifests and calculations supporting the research notes |
| `artifacts/main/smoke/` | Short diagnostic runs, not benchmark results |
| `artifacts/benchmarks/<name>/` | Benchmark `training/`, `evaluation/`, and `reports/` |
| `artifacts/archive/` | Historical retrieval experiments, source snapshots, and legacy outputs |
| `checkpoints/` | Main-dataset model weights (`.pt`) |
| `benchmarks/<name>/checkpoints/` | Benchmark model weights (`.pt`) |
| `data/` and `benchmarks/<name>/data/` | Datasets, embeddings, and SID arrays |

Training/evaluation scripts use these output directories by default. Benchmark
training and tokenizer scripts accept `--history-dir` independently of
`--checkpoint-dir`. Evaluation/report scripts retain their `--output` overrides.
For a new experiment, use a distinct output filename/run name; moving the existing
results does not add automatic versioning or prevent overwriting default outputs.

## Finding results

Start with the [research notebook](experiments/README.md) for the hypothesis,
protocol, and interpretation. Follow its links to the raw measurements.
`benchmarks/` contains executable pipelines and their READMEs, not new result files.

