# 🔒 src/evaluation/closed_source/

The two API models in the benchmark, each run through its provider's **Batch API** with structured
outputs. Videos are sampled at 1 fps (5 frames), base64-encoded, and sent inline. The flow is the
same for both: **chunk → upload → poll → merge → retry**.

| Folder | Model | API |
|:---|:---|:---|
| [`gpt/`](gpt/) | GPT-5.4-mini | OpenAI Batch API |
| [`gemini/`](gemini/) | Gemini | Google Batch API |

```bash
cd gpt
python run_gpt.py --mode batch     # build & submit batches (one at a time)
./cycle.sh --loop 600              # retrieve → merge → submit next retry, every 10 min
```

Each folder holds the runner (`run_gpt.py` / `run_gemini.py`), batch helpers
(`retrieve_batch.py`, `merge_results.py`, `retry_failed.py`, `cycle.sh`), the shared request
builders, and an `analysis/` subfolder that produces the enriched results (copied into the
top-level `results/` as `GPT-5.4-mini_analyzed.parquet` / `Gemini_analyzed.parquet`).

🔑 API keys live in a gitignored `.env` (copy `example.env`). Heavy artifacts — request batches,
raw results, frame caches — are gitignored; only code + small summaries are tracked. See
[`gpt/README.md`](gpt/README.md) for the full playbook (token limits, the `detail=low` billing
gotcha, etc.).
