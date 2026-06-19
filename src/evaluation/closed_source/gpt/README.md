# GPT evaluation

OpenAI-powered VQA for the `final_dataset` clips, with structured outputs
and prompt caching on the shared image prefix (many questions share the same
video).

## Setup

1. `cp example.env .env` and fill in `OPENAI_API_KEY`.
2. Activate the repo venv (it already has `openai`, `opencv-python`,
   `pydantic`, `python-dotenv`): `source ../../.venv/bin/activate`.

## 10-question pilot (live mode, results in seconds)

```
python run_gpt.py --mode live --limit 10 --image-detail low
python cost_report.py results/gpt_live_results.json
```

Output goes to `results/gpt_live_results.json`. Each item has `video`,
`question`, `evaluation`, `answer`, `history`, `latency_s`, `usage`
(input / cached / output tokens), and `custom_id` — the shape
`analysis/answer_analysis_agent.py` already parses.

## Full run (batch mode, async — nothing keeps running locally)

```
python run_gpt.py --mode batch --image-detail low
# …go do something else; nothing is running on your machine…
python retrieve_batch.py                # prints per-batch status and exits
python retrieve_batch.py --wait         # or poll every 60s until completion
```

`run_gpt.py --mode batch` writes one or more `batches/jsonl/requests_NNN.jsonl`
parts (auto-chunked so each stays under OpenAI's 200 MB input-file cap; the
cap is `--max-bytes-per-file`, default 150 MB), uploads each part, creates
one batch per part, and saves `batches/batch_state.json` listing every
`batch_id` plus the full `custom_id → item` map. `retrieve_batch.py` walks
the whole list and merges all batch outputs into `results/gpt_batch_results.json`.

Expected at `--image-detail low` on this dataset: roughly 26 batches
totalling ~5 GB of input JSONL. Individual batches typically finish in
minutes to a few hours (SLA 24 h) and run in parallel on OpenAI's side.

## Image detail

`--image-detail low` uses one 85-token thumbnail per frame (≈425 input
tokens for 5 frames) — cheapest, fine for scene questions. `--image-detail
high` keeps tiles at native resolution (~425 tokens per 1280×704 frame,
2125 for 5 frames) — necessary for fine-grained artifact detection.

## Prompt caching

System prompt + the 5 video frames come first in every request; the
question text comes last. Since ~4.6 questions share each video, OpenAI's
automatic prompt cache hits on the image prefix for the 2nd–Nth question
per video. The `usage.cached_input_tokens` field in the results shows how
much actually hit cache.
