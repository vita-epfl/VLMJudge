"""
Post-process MISSING answers from v4 agent results.
Extracts answers from raw VLM output where the agent reasoned correctly
but failed to call the final_answer tool.
"""
import json
import re
import sys
import argparse
from collections import Counter
from pathlib import Path


def extract_answer_from_raw(raw_text: str, question: str) -> str | None:
    """Try to extract an answer from raw VLM output."""
    if not raw_text:
        return None

    # Pattern 1: answer="..." or "answer": "..."
    m = re.search(r'["\']?answer["\']?\s*[:=]\s*["\']([^"\']+)["\']', raw_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Pattern 2: "Answer: X" or "Final Answer: X"
    m = re.search(r'(?:final\s+)?answer\s*:\s*([^\n,.]{1,30})', raw_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Determine expected answer type from question
    q_lower = question.lower()

    # Score questions (1-3)
    if 'score that ranges from 1 to 3' in q_lower or 'the answer is one of the following : 1 2 3' in q_lower:
        # Look for standalone digit 1, 2, or 3 near end of text
        # Check for "score of X", "I would give X", "rating: X"
        m = re.search(r'(?:score|rating|give)[:\s]+(\d)', raw_text, re.IGNORECASE)
        if m and m.group(1) in ('1', '2', '3'):
            return m.group(1)
        # Last digit in text
        digits = re.findall(r'\b([123])\b', raw_text[-200:])
        if digits:
            return digits[-1]

    # Yes/No questions
    if any(x in q_lower for x in ['is it a safe', 'is the ego car', 'has the ego car', 'are the traffic']):
        # Look for clear Yes/No near the end
        last_chunk = raw_text[-300:].lower()
        yes_count = len(re.findall(r'\byes\b', last_chunk))
        no_count = len(re.findall(r'\bno\b', last_chunk))
        if yes_count > no_count:
            return 'Yes'
        elif no_count > yes_count:
            return 'No'

    # Real/Generated
    if 'real video or generated' in q_lower or 'real or generated' in q_lower:
        last_chunk = raw_text[-300:].lower()
        if 'generated' in last_chunk and 'real' not in last_chunk[-100:]:
            return 'Generated'
        elif 'real' in last_chunk and 'generated' not in last_chunk[-100:]:
            return 'Real'
        # Check for stronger signals
        gen_count = last_chunk.count('generated')
        real_count = last_chunk.count('real')
        if gen_count > real_count:
            return 'Generated'
        elif real_count > gen_count:
            return 'Real'

    # Left/Right/Not overtaking
    if 'which side' in q_lower and 'overtaking' in q_lower:
        last_chunk = raw_text[-300:].lower()
        if 'not overtaking' in last_chunk or "it's not overtaking" in last_chunk:
            return "It's not overtaking"
        elif 'left' in last_chunk and 'right' not in last_chunk[-100:]:
            return 'Left'
        elif 'right' in last_chunk and 'left' not in last_chunk[-100:]:
            return 'Right'

    return None


def postprocess(input_path: str, output_path: str = None):
    with open(input_path) as f:
        data = json.load(f)

    if output_path is None:
        output_path = input_path.replace('.json', '_postprocessed.json')

    total = len(data)
    missing_count = sum(1 for e in data if e['answer'] == 'MISSING')
    recovered = 0
    still_missing = 0

    for entry in data:
        if entry['answer'] != 'MISSING':
            continue

        raw = entry.get('evaluation', '')
        # Strip the prefix
        raw = raw.replace('final_answer not called. Raw output: ', '')
        raw = raw.replace('final_answer tool was not called (no output)', '')
        raw = raw.replace('final_answer tool was not called', '')

        extracted = extract_answer_from_raw(raw, entry['question'])
        if extracted:
            # Normalize
            extracted = extracted.strip().capitalize()
            # Fix common issues
            if extracted.lower() in ('yes', 'no'):
                extracted = extracted.capitalize()
            entry['answer'] = extracted
            entry['answer_source'] = 'postprocessed'
            recovered += 1
        else:
            still_missing += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Stats
    answers = Counter(e['answer'] for e in data)
    print(f"\n=== Post-processing Results ===")
    print(f"Total entries: {total}")
    print(f"Originally MISSING: {missing_count}")
    print(f"Recovered: {recovered}")
    print(f"Still MISSING: {still_missing}")
    print(f"Recovery rate: {recovered*100/missing_count:.1f}%")
    print(f"\nAnswer distribution:")
    for k, v in answers.most_common():
        print(f"  {k}: {v} ({v*100//total}%)")
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to results JSON')
    parser.add_argument('-o', '--output', help='Output path (default: *_postprocessed.json)')
    args = parser.parse_args()
    postprocess(args.input, args.output)
