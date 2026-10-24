# Repository and result layout

Run commands from the repository root. Source files, notes, checkpoints, and outputs are kept separate.

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

Scripts use these paths by default. For benchmarks, `--history-dir` and `--checkpoint-dir` are separate; evaluation/report scripts also accept `--output`. Give each run a unique name to avoid overwriting results.

## Finding results

Start at the [research notebook](experiments/README.md) and follow links to protocols, results, and raw files. Executable benchmark pipelines live in `benchmarks/`.
