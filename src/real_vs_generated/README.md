# 🧮 real_vs_generated/

A self-contained **real-vs-AI-generated** ablation — the controlled experiment behind the headline
"models just say *real*" finding. 100 clips: **50 real 🟢 / 50 generated 🔴** (deterministic, seed 42).

| File / dir | What it is |
|:---|:---|
| `build_realgen_dataset.py` | builds the 100-clip subset + its question JSONs |
| `groundtruth_realgen.json` | the Real / Generated label per clip |
| `scripts/` | surgical copies of the eval + agent scripts (patched to accept an absolute `--questions_file`) + `startup_realgen_agent.sh` |
| `analyze_realgen.py` | computes per-class (Real vs Generated) accuracy |
| `results/` | one JSON of model answers per run |
| `realgen_accuracy_summary.json` | the final per-model Real/Generated accuracy table |
| `commands.txt` | the exact RunAI pod commands to reproduce the whole sweep |

**The result:** base VLMs score ~96–100 % on real clips but **0–22 % on generated ones**; the
Qwen3-VL **agent** reaches **64–70 %** on generated clips. See the [root README](../../README.md);
the cluster launcher `cluster/run_job_realgen.sh` is kept locally.
