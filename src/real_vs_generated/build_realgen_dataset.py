"""Build the 100-video real/generated ablation dataset.

Samples (seed-fixed):
  - 25 generated videos uniformly from cosmos-drive-dreams (excluding the
    blacklisted nOP1blfMCTg_48624_14b id, matching prep_data.py).
  - 25 generated videos uniformly from cosmos-predict1.
  - 50 real videos: all *.mp4 found under
    /mnt/vita/scratch/.../final_dataset/real (or --real_dir).

Writes two message JSONs in the same chat-format used by every eval script in
this repo (`{"role":"user","content":[{"type":"video","video":<abs>},
{"type":"text","text":<prompt>}]}`):
  - questions_realgen.json        : prompt without hints (prep_data.py shape)
  - questions_realgen_hints.json  : prompt with the 3 reality-hint questions
                                     (prep_data_with_hints.py shape)

Plus a side-car `groundtruth_realgen.json` with one entry per video for
post-processing accuracy.
"""

import argparse
import json
import os
import random
from pathlib import Path

# Question text and reality hints — taken verbatim from the canonical files
# (dataset/dataset_final/questions_cosmos-*.json and data_preparation/resources/hint_questions.json)
QUESTION = "Is it a real video or generated ?"
REALITY_HINTS = [
    "Do the shadows cast by cars and objects match the direction of the light sources?",
    "Is the layout of the road normal ?",
    "Do road markings remain consistent in width and spacing?",
]

# Same prompt shapes that prep_data.py / prep_data_with_hints.py use
GENERATED_PROMPT = """You will be given a video and a question couple.
    The question is here to evaluate your performances.
    Your task is to analyze the video and then answer by Real or Generated to the question.

    Provide your feedback as follows:

    Feedback:::
    Evaluation: (your rationale for the question, as a text)
    Answer: (your answer, as a Real or a Generated)

    You MUST provide values for 'Evaluation:' and 'Answer:' in your answer.

    Now here is the question.

    Question: {question}

    Provide your feedback."""

GENERATED_PROMPT_HINTS = """You will be given a video and a main question. To ensure accuracy, you are also provided with a set of "Hint Questions" that you must answer first.

    Your task is to analyze the video and follow this two-step reasoning process:
    1. Answer the Hint Questions to gather objective evidence from the video.
    2. Use those answers to determine the final answer to the Main Question which should either be Real or Generated.

    Hint Questions:
    {secondary_questions}

    Provide your feedback as follows:

    Feedback:::
    Evaluation: (First, write the answer to each Hint Question. Then, explain your reasoning for the Main Question based on those facts.)
    Answer: (your answer, as a Real or a Generated)

    You MUST provide values for 'Evaluation:' and 'Answer:' in your answer.

    Now here is the Main Question.

    Question: {question}

    Provide your feedback."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--storage_path", type=str,
                   default="/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset",
                   help="Absolute path to final_dataset on the cluster")
    # NOTE: source lists are derived by globbing *.mp4 directly under
    # <storage_path>/cosmos-drive-dreams and <storage_path>/cosmos-predict1.
    # No per-source question JSON is required.
    p.add_argument("--real_dir", type=str, default=None,
                   help="Override path to the 'real' folder (default: <storage_path>/real)")
    p.add_argument("--n_per_generated_source", type=int, default=25)
    p.add_argument("--n_real", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="realgen_ablation",
                   help="Directory where the message JSONs are written")
    return p.parse_args()


def build_message(video_abs_path: str, prompt: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "video", "video": video_abs_path},
            {"type": "text", "text": prompt},
        ],
    }


def sample_generated(source_subdir: str, storage_path: str,
                     n: int, rng: random.Random) -> list[dict]:
    """Sample n generated videos by directly rglob-ing *.mp4 under
    <storage_path>/<source_subdir>. Skips the same blacklisted id as
    prep_data.py.
    """
    src_root = Path(storage_path) / source_subdir
    if not src_root.exists():
        raise FileNotFoundError(f"Source folder not found: {src_root}")
    candidates = sorted([
        p for p in src_root.rglob("*.mp4")
        if p.is_file() and "nOP1blfMCTg_48624_14b" not in p.name
    ])
    if len(candidates) < n:
        raise ValueError(f"Need {n} videos under {src_root}, found {len(candidates)}")
    picked = rng.sample(candidates, n)
    out = []
    for p in picked:
        rel_video = p.relative_to(src_root).as_posix()
        out.append({"video": str(p), "source": source_subdir, "ground_truth": "Generated",
                    "rel_video": f"{source_subdir}/{rel_video}"})
    return out


def list_real(real_dir: Path, n: int, rng: random.Random) -> list[dict]:
    """List up to n *.mp4 files under real_dir. If more than n exist, sample n."""
    if not real_dir.exists():
        # Try anyway — the script may be running off-cluster; fall back to
        # treating the path as the cluster-side path and emit `n_real` synthetic
        # placeholders is NOT acceptable. Instead, raise so the user notices.
        raise FileNotFoundError(
            f"Real folder not found at {real_dir}. "
            "Pass --real_dir or run this on the cluster pod where /mnt/vita is mounted."
        )
    files = sorted([p for p in real_dir.rglob("*.mp4") if p.is_file()])
    if len(files) < n:
        raise ValueError(f"Need {n} real videos, only found {len(files)} under {real_dir}")
    picked = files if len(files) == n else rng.sample(files, n)
    out = []
    for p in picked:
        out.append({"video": str(p), "source": "real", "ground_truth": "Real",
                    "rel_video": f"real/{p.relative_to(real_dir).as_posix()}"})
    return out


def main():
    args = parse_args()
    real_dir = Path(args.real_dir) if args.real_dir else Path(args.storage_path) / "real"

    rng = random.Random(args.seed)

    gen_dd = sample_generated("cosmos-drive-dreams",
                              args.storage_path, args.n_per_generated_source, rng)
    gen_p1 = sample_generated("cosmos-predict1",
                              args.storage_path, args.n_per_generated_source, rng)
    real_videos = list_real(real_dir, args.n_real, rng)

    all_videos = gen_dd + gen_p1 + real_videos
    print(f"Generated cosmos-drive-dreams : {len(gen_dd)}")
    print(f"Generated cosmos-predict1     : {len(gen_p1)}")
    print(f"Real                          : {len(real_videos)}")
    print(f"Total                         : {len(all_videos)}")

    # Build the four message lists.
    # Baseline variants use the full Feedback::: template (downstream regex
    # parser depends on it). Agent variants ship only the bare question (and
    # hints when applicable): the agent's system prompt already mandates the
    # SOP and a terminal `final_answer` tool call — adding the inline
    # Feedback template here makes the agent follow that inline rubric
    # instead of calling the tool, breaking the eval shape.
    prompt_no_hint = GENERATED_PROMPT.format(question=QUESTION)
    formatted_hints = "\n".join([f"- {h}" for h in REALITY_HINTS])
    prompt_with_hint = GENERATED_PROMPT_HINTS.format(question=QUESTION,
                                                     secondary_questions=formatted_hints)
    agent_prompt = QUESTION
    agent_prompt_hints = (
        f"{QUESTION}\n\n"
        "Hint Questions to consider as supporting evidence:\n"
        f"{formatted_hints}"
    )

    msgs_no_hint = [build_message(v["video"], prompt_no_hint) for v in all_videos]
    msgs_with_hint = [build_message(v["video"], prompt_with_hint) for v in all_videos]
    msgs_agent = [build_message(v["video"], agent_prompt) for v in all_videos]
    msgs_agent_hints = [build_message(v["video"], agent_prompt_hints) for v in all_videos]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    no_hint_path = out_dir / "questions_realgen.json"
    with_hint_path = out_dir / "questions_realgen_hints.json"
    agent_path = out_dir / "questions_realgen_agent.json"
    agent_hints_path = out_dir / "questions_realgen_agent_hints.json"
    gt_path = out_dir / "groundtruth_realgen.json"

    with open(no_hint_path, "w") as f:
        json.dump(msgs_no_hint, f, indent=4, ensure_ascii=False)
    with open(with_hint_path, "w") as f:
        json.dump(msgs_with_hint, f, indent=4, ensure_ascii=False)
    with open(agent_path, "w") as f:
        json.dump(msgs_agent, f, indent=4, ensure_ascii=False)
    with open(agent_hints_path, "w") as f:
        json.dump(msgs_agent_hints, f, indent=4, ensure_ascii=False)
    with open(gt_path, "w") as f:
        json.dump(all_videos, f, indent=4, ensure_ascii=False)

    print(f"\nWrote {no_hint_path}   ({len(msgs_no_hint)} messages, baseline no-hints)")
    print(f"Wrote {with_hint_path}   ({len(msgs_with_hint)} messages, baseline hints)")
    print(f"Wrote {agent_path}   ({len(msgs_agent)} messages, agent bare)")
    print(f"Wrote {agent_hints_path}   ({len(msgs_agent_hints)} messages, agent bare + hints)")
    print(f"Wrote {gt_path}                 ({len(all_videos)} ground-truth records)")


if __name__ == "__main__":
    main()
