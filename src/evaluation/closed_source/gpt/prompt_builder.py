"""Build chat messages with a cacheable prefix.

Layout (ordered):
    system      : constant across *every* request          ─┐
    user[0..N-1]: the N frames of the video                 │ cached prefix
                                                            ┘
    user[N]     : question text + allowed answers           — variable suffix

OpenAI prompt-caching is prefix-based: keeping image tokens identical across
requests for the same video, and putting them *before* the question text, is
the whole point. We pack the frames in one user message followed by the
question text in the same user message's content list — that preserves token
order `system → images → text` which is what the cache keys on.
"""

from __future__ import annotations

from typing import Literal

from frame_extractor import Frame
from schemas import _Base, allowed_answers

ImageDetail = Literal["low", "high", "auto"]

SYSTEM_PROMPT = (
    "You are an expert video analyst specialised in driving scenes and "
    "AI-generation artifact detection. You will be given frames sampled at "
    "1 frame per second from a short driving clip, followed by a question. "
    "Analyse the frames carefully and return a structured answer: a 1–3 "
    "sentence rationale in `evaluation`, and the final choice in `answer` "
    "(which must match one of the allowed options exactly)."
)


def build_messages(
    frames: list[Frame],
    question: str,
    schema: type[_Base],
    *,
    image_detail: ImageDetail = "low",
) -> list[dict]:
    options = allowed_answers(schema)
    image_parts = [
        {
            "type": "image_url",
            "image_url": {"url": f.data_url, "detail": image_detail},
        }
        for f in frames
    ]
    question_text = (
        f"Question: {question}\n"
        f"Allowed answers: {options}\n"
        "Respond using the structured schema."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                *image_parts,
                {"type": "text", "text": question_text},
            ],
        },
    ]
