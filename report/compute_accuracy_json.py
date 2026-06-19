"""Compute per-model accuracy + mean inference time and emit two JSON files.

- models-accuracy.json: all timed models + GPT-5.4-mini
- accuracy-vs-time.json: all timed models (GPT excluded)

For the base-model timing parquets, `vlm_answer` still contains the full raw
"Feedback:::...Answer: X" output — we extract the answer the same way
analysis/answer_analysis.py does before comparing to `ground_truth`.
Agent parquets (v4o, v4p) already store a clean vlm_answer, and GPT's parquet
already has a `correct` column.
"""
import json
import re
from pathlib import Path

import polars as pl

RESULTS_DIR = Path("/home/matthieu/Documents/Projet de master/VLM-eval-rcp/to_be_copied/results")
OUT_DIR = Path("/home/matthieu/Documents/Projet de master/VLM-eval-rcp")

# file stem -> (display name, category, needs_answer_extraction)
MODEL_MAP = {
    "Qwen3VL_analyzed":          ("Qwen3-VL",         "base",  True),
    "Qwen_omni_analyzed":        ("Qwen-Omni",        "base",  True),
    "llava_onevision_analyzed":  ("LLaVA-OneVision",  "base",  True),
    "Cosmos_Reason_analyzed":    ("Cosmos-Reason",    "base",  True),
    "InternVL3_5-8B_analyzed":   ("InternVL3.5-8B",   "base",  True),
    "InternVL3_5-30B_analyzed":  ("InternVL3.5-30B",  "base",  True),
    "v4o_analyzed":              ("Agent v4o",        "agent", False),
    "v4p_analyzed":              ("Agent v4p",        "agent", False),
}

ANSWER_RE = re.compile(r"Answer:\s*(.*?)(?:\s*<\|im_end\|>|$|\n)", re.DOTALL)


def extract_answer(raw: str) -> str:
    if raw is None:
        return ""
    m = ANSWER_RE.search(raw)
    return m.group(1).strip().capitalize() if m else ""


accuracy_entries = []
point_entries = []

for stem, (display_name, category, needs_extract) in MODEL_MAP.items():
    df = pl.read_parquet(RESULTS_DIR / f"{stem}.parquet")

    if needs_extract:
        parsed = [extract_answer(x) for x in df["vlm_answer"].to_list()]
        gt = df["ground_truth"].to_list()
        correct_list = [p == g for p, g in zip(parsed, gt)]
        acc = sum(correct_list) / len(correct_list) * 100
    else:
        acc = df["correct"].cast(pl.Int64).mean() * 100

    mean_time = df["inference_time"].mean()

    accuracy = round(acc, 1)
    time_s = round(mean_time, 2)

    accuracy_entries.append({"name": display_name, "accuracy": accuracy, "highlight": False})
    point_entries.append({
        "name": display_name,
        "time": time_s,
        "accuracy": accuracy,
        "category": category,
        "highlight": False,
    })

gpt = pl.read_parquet(RESULTS_DIR / "gpt_results_enriched.parquet")
gpt_acc = round(gpt["correct"].cast(pl.Int64).mean() * 100, 1)
accuracy_entries.append({"name": "GPT-5.4-mini", "accuracy": gpt_acc, "highlight": False})

accuracy_entries.sort(key=lambda e: e["accuracy"])
point_entries.sort(key=lambda e: e["accuracy"])

(OUT_DIR / "models-accuracy.json").write_text(json.dumps({
    "title": "Accuracy of the models",
    "models": accuracy_entries,
}, indent=2) + "\n")

(OUT_DIR / "accuracy-vs-time.json").write_text(json.dumps({
    "xLabel": "Inference time (s)",
    "yLabel": "Accuracy (%)",
    "points": point_entries,
}, indent=2) + "\n")

print("models-accuracy.json:")
for e in accuracy_entries:
    print(f"  {e['name']:20s}  {e['accuracy']:>5.1f}%")
print("\naccuracy-vs-time.json:")
for p in point_entries:
    print(f"  {p['name']:20s}  t={p['time']:>7.2f}s  acc={p['accuracy']:>5.1f}%  [{p['category']}]")
