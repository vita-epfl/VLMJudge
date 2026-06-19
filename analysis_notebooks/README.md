# 📓 analysis_notebooks/

The analysis story in charts. Each notebook reads the cleaned outputs produced by
[`../src/analysis/`](../src/analysis/) and the parquets in `../dataset/` / `../results/`.

| Notebook | Shows |
|:---|:---|
| `benchmark_analysis.ipynb` | overall + per-category accuracy across all models & agents |
| `benchmark_timing.ipynb` | accuracy ↔ inference-time trade-off |
| `agentic_failure_analysis.ipynb` | agent failure modes (tool misuse, missing answers) |
| `questions_analysis.ipynb` | question diversity & difficulty |

> ℹ️ These were moved here from the repo root during cleanup; their relative data paths point at
> `../dataset/…` and figures are written to `../report/figures/…`. **Run them from this directory**
> (or launch Jupyter here) so the paths resolve.

See the [root README](../README.md).
