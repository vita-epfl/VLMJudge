# 📊 src/analysis/

Post-processing of the raw results JSON into clean, ground-truth-joined accuracy tables (polars).

Each script regex-extracts the `Question:` / `Answer:` markers from raw VLM output and joins against
the ground truth on `(video, question)`:

- `answer_analysis.py` — the base cleaner for open-source model outputs.
- `answer_analysis_agent.py` — for agent / closed-source outputs (already structured, no extraction).
- `answer_analysis_timing.py` — keeps the latency columns from the `eval_*_timing.py` runs.
- `classify_v2.py` — cue/category classification → `cue_summary_v2.json`.

> ⚠️ The parsing regex is coupled to the prompt's fixed `Feedback:::\nEvaluation: …\nAnswer: …`
> output shape — don't change one without the other.

Outputs are consumed by the [`../../analysis_notebooks/`](../../analysis_notebooks/). See the [root README](../../README.md).
