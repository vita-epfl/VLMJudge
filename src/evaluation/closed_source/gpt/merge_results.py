"""Merge multiple result JSONs into one, deduplicating by custom_id.

When the same custom_id appears in multiple files, the *last* file wins.
This means: pass the original results first, retries second — retry
successes replace original errors.

Usage:
    python merge_results.py \\
        results/gpt_batch_results.json \\
        results/gpt_retry_results.json \\
        -o results/gpt_all_results.json

    # After a second retry round, just append the new file:
    python merge_results.py \\
        results/gpt_all_results.json \\
        results/gpt_retry2_results.json \\
        -o results/gpt_all_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="Result JSON files, in priority order (last wins).")
    ap.add_argument("-o", "--output", type=Path, required=True,
                    help="Where to write the merged result.")
    args = ap.parse_args()

    merged: dict[str, dict] = {}
    total_loaded = 0
    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        total_loaded += len(items)
        for item in items:
            cid = item.get("custom_id")
            if cid is None:
                continue
            merged[cid] = item

    results = list(merged.values())
    ok = sum(1 for r in results if r.get("answer") and not r.get("error"))
    err = sum(1 for r in results if r.get("error") or not r.get("answer"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Loaded {total_loaded} items from {len(args.inputs)} file(s).")
    print(f"After dedup by custom_id: {len(results)} unique items.")
    print(f"  OK     : {ok}")
    print(f"  Errored: {err}")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
