# 📦 src/data_preparation/

Turns the raw parquet datasets + MP4 videos into JSON files of **chat-formatted messages**, one per
`(video, question)` pair, ready for the evaluation scripts. All write into `dataset/`.

- `prep_data.py` — the canonical builder → `Questions.json` (the main benchmark question set, used
  by every `../evaluation/eval_*.py`).
- `prep_data_with_hints.py` — `Questions_with_hint.json` (adds the hint pool from
  `resources/hint_questions.json`; used by the hinted agent runs).
- `prep_data_direct_prompting.py` — `Questions_raw.json` (minimal prompt; used by the v4 agent).

See the [root README](../../README.md) for how this feeds the pipeline.
