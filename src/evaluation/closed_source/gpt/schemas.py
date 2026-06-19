"""Structured-output schemas, one per answer family.

Mirrors the two-field contract of the agentic `final_answer` tool:
    {"evaluation": <rationale>, "answer": <one of a closed set>}
`answer` is `Literal[...]` so OpenAI's `strict` structured-output mode refuses
anything off-menu and `vlm_answer` in downstream analysis lands already
normalised.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field


class _Base(BaseModel):
    evaluation: str = Field(
        ...,
        description="Your detailed rationale based on the provided frames (1–4 sentences).",
    )


class YesNoAnswer(_Base):
    answer: Literal["Yes", "No"]


class RealGeneratedAnswer(_Base):
    answer: Literal["Real", "Generated"]


class ScoreAnswer(_Base):
    answer: Literal["1", "2", "3"]


class OvertakeAnswer(_Base):
    answer: Literal["Left", "Right", "It's not overtaking"]


# The 17 unique question strings in Questions_raw.json were manually inspected
# and grouped by expected answer family. We match after normalising whitespace
# and stripping the "The answer is one of the following: ..." suffix that the
# original prep scripts appended to some MCQs.

_SUFFIX_RE = re.compile(
    r"\s*The answer is one of the following\s*:.*",
    flags=re.IGNORECASE | re.DOTALL,
)


def _normalise(question: str) -> str:
    return _SUFFIX_RE.sub("", question).strip()


_EXACT_MAP: dict[str, type[_Base]] = {
    # Real vs Generated
    "Is it a real video or generated ?": RealGeneratedAnswer,
    # Overtake side
    "On which side is the ego car overtaking ?": OvertakeAnswer,
    # MCQ 1–3
    ("Overall, how realistic is the scene (roads, markings, lighting, objects) ? "
     "Give a score that ranges from 1 to 3 with the following scale : "
     "1=clearly impossible, 2=mostly realistic with minor issues, "
     "3=indistinguishable from real"): ScoreAnswer,
    ("Overall, how safe is the driving of the ego car ? "
     "Give a score that ranges from 1 to 3 with the following scale : "
     "1=really dangerous, 2=mostly safe with minor issues, 3=perfectly safe"): ScoreAnswer,
}

# Everything else in the raw file is a Yes/No.
_YES_NO_QUESTIONS = {
    "Is it a safe way of driving for the ego car ?",
    "Is the ego car on a highway ?",
    "Is the ego car following the traffic rules ?",
    "Is the ego car behavior consistent with visible cues of the traffic lights ?",
    "Are the traffic lights red ?",
    "Are the traffic signs consistent with ego movement? For example, after turning.",
    "Has the ego car stopped ?",
    "Has the ego car stopped after the sign ?",
    "Has the ego car stopped after the stop ?",
    "Has the ego car stopped after the stop sign ?",
    "Is the lighting changing ?",
    "Does any object appear or disappear without continuous motion ?",
    "Do some objects change shape or appearance ?",
}


def schema_for(question: str) -> type[_Base]:
    norm = _normalise(question)
    if norm in _EXACT_MAP:
        return _EXACT_MAP[norm]
    if norm in _YES_NO_QUESTIONS:
        return YesNoAnswer
    # Heuristic fallbacks if a new question slips in.
    if "real or generated" in norm.lower() or "real video or generated" in norm.lower():
        return RealGeneratedAnswer
    if norm.lower().startswith("on which side is the ego car overtaking"):
        return OvertakeAnswer
    if re.search(r"score that ranges from 1 to 3", norm, flags=re.IGNORECASE):
        return ScoreAnswer
    raise ValueError(f"No schema registered for question: {question!r}")


def allowed_answers(schema: type[_Base]) -> list[str]:
    """Human-readable list of the `Literal` values for prompt construction."""
    # `answer` field's annotation is Literal[...] — extract its args.
    field = schema.model_fields["answer"]
    return list(field.annotation.__args__)  # type: ignore[union-attr]


if __name__ == "__main__":
    import json
    from path_fix import load_questions

    items = load_questions("./dataset/dataset_final/Questions_raw.json")
    counts: dict[str, int] = {}
    for it in items:
        counts[schema_for(it.question).__name__] = counts.get(
            schema_for(it.question).__name__, 0
        ) + 1
    print(json.dumps(counts, indent=2))
