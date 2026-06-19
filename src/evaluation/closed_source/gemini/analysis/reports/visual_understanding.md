# Visual understanding — summary stats

- **Accuracy**: 291/304 = **95.7%** (Gemini 3 Flash Preview, 5 JPEG frames @ 1 fps).
- **Question mix**: only three prompt templates appear.
  - `Is the ego car on a highway ?` — 159 (138 Yes / 21 No)
  - `Are the traffic lights red ?` — 144 (138 Yes / 6 No)
  - `Is the lighting changing ?` — 1 (Yes)
- **Per-cell accuracy** (the headline diagnostic):
  - highway = Yes: **138/138 (100%)**
  - highway = No: **16/21 (76%)** — 5 errors
  - red = Yes: **136/138 (98.6%)** — 2 errors
  - red = No: **0/6 (0%)** — 6 errors
- **Evaluations are short and decisive**: mean 205 chars, median 196, range 113–346. One paragraph, 1–2 sentences, almost always a list of perceptual cues followed by a verdict. No tool reasoning, no hedging.

# What drives near-perfect accuracy (brief, with quotes)

The benchmark's two dominant questions have unambiguous, high-contrast visual signatures, and the GT prior is ~95% "Yes" on both. Gemini latches onto a small canonical cue list and that is exactly what's typically present:

- Highway: multi-lane, divider/guardrail, green overhead signs, semi-trucks, no intersections.
  > "multi-lane road with overhead green directional signs and a guardrail, which are typical characteristics of a highway" (`agg_tk_003_Original`)
  > "lack of intersections or cross-traffic further confirm it is a highway" (`agg_tk_007_Rainy`)
- Red lights: any visible red bulb facing the camera at the upcoming intersection.
  > "multiple red traffic lights at the upcoming intersection, both overhead and on the sides" (`cross_red_033_Original`)

When those cues match the (frequent) Yes label, Gemini scores 100% / 98.6%. The single "Is the lighting changing?" item is a free hit:
> "significant increase in brightness from the first frame to the final frame… clear change in lighting conditions" (`car_crash_010_Rainy`).

# The 13 failures — why? (detailed, with quotes)

The 13 errors split cleanly into two failure modes, both **dominated by minority-class items**: the model essentially never returns the rare answer.

### Mode A — "Yes-bias on red-light questions" (8/13 errors, all 6 GT=No items + 2 GT=Yes)
For `cross_red_*` clips labelled No, Gemini still sees a red bulb somewhere in the scene and answers Yes. It does not distinguish red lights for the ego direction vs. cross-traffic / left-turn-only / secondary signals. Six straight failures, all with confident wording:
> `cross_red_038_Night` (GT=No): "multiple traffic lights ahead showing red signals at the intersection as the vehicle approaches and stops."
> `cross_red_009_Rainy` (GT=No): "traffic lights facing the camera remain red across the first four frames… clearly visible despite the rainy conditions."
> `cross_red_010_Golden hour` (GT=No): "several traffic lights are clearly visible and displaying a red signal… consistent indication throughout the sequence is that the lights are red."
> `cross_red_025_Rainy` (GT=No): "the traffic lights visible at the upcoming intersection are displaying a red signal as the vehicle approaches."

The two GT=Yes red-light misses are the inverse error — when the ego-direction signal happens to be red but other green signals are present, Gemini votes green:
> `cross_red_007_Original` (GT=Yes): "While there are some red lights visible on the overhead mast arms (likely for left turns or secondary signals), the main signals for through traffic are green."
> `cross_red_017_Original` (GT=Yes): "a single red light is visible on an overhead mast in the first two frames, the vehicle passes it and approaches an intersection where all visible traffic signals are clearly green."

Both inverse cases show the model *could* reason about which signal applies, but it disagrees with the labeller — these may be borderline GT.

### Mode B — "Highway-bias on aggressive-takeover clips" (5/13 errors, all GT=No)
All five misses are `aggressive_takeover/agg_tk_*` clips where the model fixates on highway furniture (semi-trucks, green signage, sound walls, guardrails) and ignores whatever feature actually disqualifies it as a highway. The wording is identical across the five clips:
> `agg_tk_032_Snowy` (GT=No): "multiple lanes in each direction, a central divider with guardrails, and large green signage… presence of multiple semi-trucks further supports that this is a highway environment."
> `agg_tk_023_Original` (GT=No): "multi-lane road at night with dashed white lane markings and the presence of large commercial semi-trucks. The lack of intersections, traffic lights, and pedestrians… indicates a highway environment."
> `agg_tk_022_Original` (GT=No): "concrete median barrier on the left… large commercial trucks and green highway signage further indicates… on a highway."

Because GT here is "No", these clips presumably contain a non-highway feature (intersection further along, urban transition, arterial road that merely *looks* highway-like). Gemini's 5-frame, 1 fps view doesn't catch it — the cue list is sufficient at the frame level even when the full clip semantics would say otherwise.

# Hypotheses

1. **Class-prior contamination dominates the residual errors.** GT distribution is ~95% Yes on both leading questions; 11/13 misses are minority-class items. Gemini behaves close to "always Yes" plus a few high-confidence Nos, which is near-optimal under this prior.
2. **No part-of-scene reasoning for traffic lights.** The model does not separate "red light governing the ego lane" from "red bulb visible anywhere in the frame" — every red-light No is justified by *some* red bulb being present. This is consistent with single-frame, no-tool perception.
3. **Highway = static cue dictionary.** Five identically-phrased highway misses suggest the model has a fixed checklist (lanes / divider / green signs / trucks / no intersections) and votes Yes whenever ≥3 cues fire, regardless of clip-level context.
4. **Evaluations are confident, terse, single-paragraph rationalisations** — no uncertainty, no per-frame disagreement, no tool calls. The format itself probably doesn't help on edge cases: there is no place where the model second-guesses the cue list.
5. **Likely ceiling unless the test set is rebalanced.** Adding more `No` items (or harder cross-traffic-only red-light clips) would expose Mode A and Mode B and probably drop accuracy 5–10 points without otherwise changing the model.
