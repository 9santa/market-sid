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
The old top-level `experiments/` directory has been split between documentation
and `artifacts/main/derived/`.

## Historical paths and integrity

The cleanup on 2026-10-21 moved files without rewriting historical JSON, NPZ,
logs, or archived source snapshots. Model weights and data stayed in place.
Markdown links and command examples now point to the new locations.

[The migration manifest](repository_layout_migration.json) records every old path,
new path, and SHA-256 before the move. Old paths embedded in evidence JSON or
checkpoint metadata remain unchanged so their hashes remain verifiable.
For example, a saved `training_history` or `per_user_results.path` may use an old path.
Resolve it when reading historical metadata:

```python
import json
import numpy as np
from src.result_paths import resolve_result_path

result = json.loads(resolve_result_path(
    "artifacts/prefix_retrieval_valid_full_tt_control.json"
).read_text())
rows = np.load(resolve_result_path(result["per_user_results"]["path"]))
```

The SID comparison report resolves old training-history references automatically.
The main model comparison report also accepts old input result paths. Other
commands and external notebooks should use the new paths or this helper explicitly.
Archived code is preserved as evidence; its embedded paths have not been modernized.

Raw outputs, weights, and data remain ignored by Git. Research Markdown, migration
metadata, small derived evidence, and the three previously tracked baseline
histories are included. The local `docs/notes.md` scratchpad remains ignored.
