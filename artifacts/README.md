# Results and artifacts

- `main/`: main-dataset training histories, evaluations, diagnostics, and reports.
- `benchmarks/<name>/`: training histories, evaluations, and reports for each benchmark.
- `archive/`: preserved historical experiments and source snapshots.

See the [layout guide](../docs/repository_layout.md) for all subdirectories and
the [research notebook](../docs/experiments/README.md) for interpretations.
Weights belong in `checkpoints/`; human-written experiment notes belong in `docs/`.
Historical paths embedded inside JSON are resolved through the migration manifest;
the original measurements have not been rewritten.
