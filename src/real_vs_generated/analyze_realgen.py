"""Accuracy analysis for the real/generated ablation.

Reads every <model>_realgen.json file in results/ and joins each entry against
groundtruth_realgen.json on the video relative path. Reports:
  - Total accuracy (all 100 items)
  - Per-GT-category accuracy ("Real" vs "Generated" subsets)
  - Count of unparseable responses (counted as wrong)

Handles both result shapes:
  - Baseline shape: {video, question, answer}  where `answer` is the raw
    "Feedback:::\\nEvaluation: …\\nAnswer: X" text emitted by the prompt
    template. The final answer is extracted by taking the LAST occurrence of
    `Answer:` and normalising the token to Real/Generated.
  - Agent shape (v4o / v4p style): {video, question, evaluation, answer,
    history, …} where `answer` is already the clean final token. We still
    normalise it to handle stray casing or punctuation.

Unparseable response (no Real/Generated token after normalisation) ⇒ counted
as wrong, as requested.
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/matthieu/Documents/Projet de master/VLM-eval-rcp/realgen_ablation")
RESULTS_DIR = ROOT / "results"
GT_PATH = ROOT / "groundtruth_realgen.json"

# Optional pretty model labels for known files; anything else falls back to the
# stem of the filename without the `_realgen` suffix.
MODEL_LABELS = {
    "Cosmos_Reason_realgen.json":            "Cosmos-Reason",
    "InternVL3_5-30B_realgen.json":          "InternVL3.5-30B",
    "InternVL3_5-8B_realgen.json":           "InternVL3.5-8B",
    "Qwen3VL_realgen.json":                  "Qwen3-VL",
    "Qwen3VL-hints_realgen.json":            "Qwen3-VL+hints",
    "Qwen_omni_realgen.json":                "Qwen2.5-Omni",
    "llava_onevision_realgen.json":          "LLaVA-OneVision",
    "Qwen3VL_agent_realgen.json":            "Qwen3-VL+agent",
    "Qwen3VL_agent_hints_realgen.json":      "Qwen3-VL+agent+hints",
}

ANSWER_RE = re.compile(r"Answer\s*:\s*([^\n<]+)", re.IGNORECASE)
PATH_RE = re.compile(r"final_dataset/(.+)$")


def relative_video(abs_path: str) -> str:
    """Convert /mnt/.../final_dataset/<source>/<rest>.mp4 → <source>/<rest>.mp4."""
    if abs_path is None:
        return ""
    m = PATH_RE.search(abs_path)
    return m.group(1) if m else abs_path


def normalise_token(text: str) -> str | None:
    """Return 'Real' or 'Generated' if the text contains one of those tokens
    (case-insensitive, last occurrence wins so chain-of-thought references
    don't shadow the final verdict). Otherwise None.
    """
    if not text:
        return None
    t = text.strip().strip(" .\t\r\n*\"'")
    # Whole-string match first (clean agent answers)
    low = t.lower()
    if low == "real":
        return "Real"
    if low == "generated":
        return "Generated"
    # Last keyword occurrence (handles baseline raw_output left over after
    # the Answer: regex partially matched but token was buried in chain-of-thought)
    last_real = low.rfind("real")
    last_gen = low.rfind("generated")
    if last_real == -1 and last_gen == -1:
        return None
    return "Generated" if last_gen > last_real else "Real"


def extract_answer(entry: dict) -> str | None:
    """Return the model's final Real/Generated verdict, or None if unparseable.

    Tries the agent shape first (clean `evaluation` + `answer`), then falls
    back to baseline shape (raw `answer` string with embedded Feedback markers).
    """
    raw = entry.get("answer")
    # Agent shape: short clean answer
    if isinstance(raw, str) and len(raw) <= 60 and ("evaluation" in entry or "history" in entry):
        tok = normalise_token(raw)
        if tok is not None:
            return tok
        # Agent didn't call final_answer → `answer` is "MISSING" and the raw
        # model text is stashed in `evaluation` as
        # "final_answer not called. Raw output: …answer: X…". Parse it the same
        # way as the baseline shape (last `answer:` marker, else last token).
        ev = entry.get("evaluation")
        if isinstance(ev, str):
            matches = list(ANSWER_RE.finditer(ev))
            if matches:
                tok = normalise_token(matches[-1].group(1))
                if tok is not None:
                    return tok
            return normalise_token(ev)
        return None
    # Baseline shape: pull the last "Answer:" match from the raw text
    if isinstance(raw, str):
        matches = list(ANSWER_RE.finditer(raw))
        if matches:
            return normalise_token(matches[-1].group(1))
        # No Answer: marker → scan the whole text for a final Real/Generated token
        return normalise_token(raw)
    return None


def load_gt() -> dict[str, str]:
    with open(GT_PATH) as f:
        records = json.load(f)
    return {r["rel_video"]: r["ground_truth"] for r in records}, records


def evaluate_one(path: Path, gt_map: dict[str, str]) -> dict:
    with open(path) as f:
        entries = json.load(f)
    per_gt = defaultdict(lambda: {"n": 0, "correct": 0, "unparseable": 0})
    total_n = 0
    total_correct = 0
    total_unparseable = 0
    skipped_no_gt = 0

    for e in entries:
        rel = relative_video(e.get("video", ""))
        if rel not in gt_map:
            skipped_no_gt += 1
            continue
        gt = gt_map[rel]
        pred = extract_answer(e)
        unparseable = pred is None
        # Unparseable counts as wrong per user requirement
        correct = (pred == gt) and (pred is not None)

        total_n += 1
        total_correct += int(correct)
        total_unparseable += int(unparseable)
        per_gt[gt]["n"] += 1
        per_gt[gt]["correct"] += int(correct)
        per_gt[gt]["unparseable"] += int(unparseable)

    return {
        "n": total_n,
        "correct": total_correct,
        "accuracy": (total_correct / total_n) if total_n else 0.0,
        "unparseable": total_unparseable,
        "skipped_no_gt": skipped_no_gt,
        "per_gt": {
            gt: {
                "n": s["n"],
                "correct": s["correct"],
                "accuracy": (s["correct"] / s["n"]) if s["n"] else 0.0,
                "unparseable": s["unparseable"],
            }
            for gt, s in per_gt.items()
        },
    }


def main() -> int:
    if not GT_PATH.exists():
        print(f"ERROR: ground-truth file missing at {GT_PATH}", file=sys.stderr)
        return 1
    gt_map, gt_records = load_gt()

    # Sanity: ground-truth category counts
    gt_counts = defaultdict(int)
    for r in gt_records:
        gt_counts[r["ground_truth"]] += 1
    print(f"Ground truth ({len(gt_records)} videos): "
          + ", ".join(f"{k}={v}" for k, v in sorted(gt_counts.items())))
    print()

    result_files = sorted([p for p in RESULTS_DIR.glob("*_realgen.json") if p.is_file()])
    if not result_files:
        print(f"No *_realgen.json files in {RESULTS_DIR}")
        return 1

    summary = {}
    print(f"{'Model':24s} {'n':>4s} {'acc':>7s} {'Real acc':>10s} {'Gen acc':>10s} {'unparse':>8s}")
    print("-" * 70)
    for path in result_files:
        label = MODEL_LABELS.get(path.name, path.stem.replace("_realgen", ""))
        try:
            r = evaluate_one(path, gt_map)
        except Exception as exc:
            print(f"{label:24s}  ERROR: {exc}")
            continue
        real = r["per_gt"].get("Real", {"n": 0, "correct": 0, "accuracy": 0.0, "unparseable": 0})
        gen = r["per_gt"].get("Generated", {"n": 0, "correct": 0, "accuracy": 0.0, "unparseable": 0})
        print(f"{label:24s} {r['n']:>4d} "
              f"{r['accuracy']*100:>6.1f}% "
              f"{real['accuracy']*100:>8.1f}% ({real['correct']}/{real['n']}) "
              f"{gen['accuracy']*100:>5.1f}% ({gen['correct']}/{gen['n']}) "
              f"{r['unparseable']:>7d}")
        summary[label] = {"file": path.name, **r}

    out_path = ROOT / "realgen_accuracy_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
