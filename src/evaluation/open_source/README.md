# 🧪 src/evaluation/open_source/

One inference script per **open-source** VLM in the benchmark. Each loads a prepared
`(video, question)` message JSON (from [`../../data_preparation/`](../../data_preparation/)), runs
**batched** inference, and writes a results JSON to the cluster `results/` dir.

| Script | Model | Backend |
|:---|:---|:---|
| `eval_qwen_3_VL.py` | Qwen3-VL-30B-A3B | vLLM |
| `eval_qwen_omni.py` | Qwen3-Omni-30B-A3B | transformers |
| `eval_cosmos_reason.py` | Cosmos-Reason-7B | transformers |
| `eval_InternVL3_5.py` | InternVL3.5-8B / 30B | lmdeploy |
| `eval_LLaVa.py` | LLaVA-OneVision-7B | vLLM |

The `*_timing.py` variants add latency instrumentation (they feed
`../../analysis/answer_analysis_timing.py` and `../../../analysis_notebooks/benchmark_timing.ipynb`).

▶️ **Tuned invocations (batch size / TP / context length) live in `cluster/commands_to_run.txt`
(kept locally) — read it first.** See the [root README](../../../README.md).
