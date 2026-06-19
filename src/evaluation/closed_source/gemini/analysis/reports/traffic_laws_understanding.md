## Traffic laws understanding — summary stats

- **Overall**: 395 / 665 correct = **59.4%** (vs Gemini's 64.0% global mean → underperforms its own average).
- **Per question**:
  - "Is the ego car following the traffic rules?" — 303/512 = 59%
  - "Is the ego car behavior consistent with visible cues of the traffic lights?" — 92/153 = 60%
- **Per video subset (where the gap really is)**:
  - `crossing_stop/` — 165/201 = **82%** correct
  - `crossing_red_lights/` — 216/304 = **71%** correct
  - `aggressive_takeover/` — 14/160 = **9%** correct
- **Confusion matrix** (GT, VLM): No/No 388, No/Yes 267, Yes/No 3, Yes/Yes 7. Ground truth is overwhelmingly "No" (655/665 violations), and Gemini's only systematic error mode is **false-negative violation calls** (267 of 270 errors are "Yes" when truth is "No"). It almost never hallucinates a violation when the car is compliant (3 false positives).

The headline finding: Gemini is excellent at the "did the ego run a red light / stop sign" sub-task, but collapses on aggressive-takeover scenarios where the violation is encoded in subtle motion (shoulder driving, lane straddling, off-road excursions across 5 frames).

## What it gets right (with quotes)

When the violation cue is **visible in a single frame** — a red light, a stop sign + STOP pavement marking — Gemini reasons cleanly and even references frame indices:

> "The ego car approaches an intersection with a clearly visible stop sign and 'STOP' pavement marking but fails to come to a complete stop. It continues to move forward through the intersection across consecutive frames" (`cross_stop_025_Rainy.mp4`)

> "The traffic lights at the intersection are clearly red throughout the sequence of frames. The ego car proceeds to drive through the intersection while the light is still red" (`cross_red_032_Snowy.mp4`)

It correctly handles the **right-turn-on-red exception**:

> "The ego car is observed making a right turn at an intersection while the traffic light is red. … the vehicle moves through the intersection and past the stop line without appearing to come to a complete stop" (`cross_red_018_Original.mp4`) — answers No (violation), correctly identifying that no full stop was made before the turn.

It does **multi-frame trajectory comparison** to infer "no stop occurred":

> "Between frames 3 and 4, the vehicle continues its forward motion and crosses the stop line without coming to a complete halt" (`cross_stop_012_Morning.mp4`)

> "the ego car continues to move from before the stop line (Frame 2) to on top of the crosswalk (Frame 3) and then into the intersection (Frame 4), failing to come to a complete stop" (`cross_stop_026_Foggy.mp4`)

It even surfaces **AI-generation artifacts** mid-reasoning:

> "By the third frame, this overhead light arm abruptly disappears (an AI artifact) and a green light is visible on a side pole, demonstrating that the car's behavior is inconsistent with the initially displayed red signal" (`cross_red_016_Sunny.mp4`)

And it handles the rare GT=Yes cases (compliant driving) without hallucinating a violation:

> "The ego car is shown driving on a multi-lane highway at night, maintaining a safe following distance … performs a standard lane change … using dashed white lines" (`agg_tk_023_Night.mp4`, GT=Yes, correct).

## What it gets wrong (with quotes)

**Aggressive-takeover (146/160 wrong, 91% error rate).** The video subset is named for the violation, but the model treats it as ordinary highway footage and rubber-stamps it "compliant":

> "The ego car is driving on a highway during foggy conditions, maintaining its lane and following the flow of traffic at a reasonable speed. No traffic violations, such as illegal lane changes or running red lights, are visible" (`agg_tk_001_Foggy.mp4`, GT=No)

> "The ego car maintains a safe following distance from vehicles ahead and stays consistently within its lane markings despite the low visibility caused by the fog" (`agg_tk_011_Foggy.mp4`, GT=No)

> "The ego car is observed driving on a multi-lane highway during low-visibility conditions. It maintains a steady course within lane markings, safely passes other vehicles like the truck on the right" (`agg_tk_036_Rainy.mp4`, GT=No)

When it does spot the issue (e.g. `agg_tk_034_Night.mp4`, `agg_tk_036_Original.mp4`), the violation is the kind that survives 1-fps sampling — drifting *across a solid line* visible in the final frame, or driving onto an unpaved shoulder. Subtle aggressive-takeover dynamics (sudden lateral acceleration, cutting in front, tailgating buildup) are wiped out by the 5-frame, 1 fps sub-sample.

**Stopped-but-still-a-violation cases.** A handful of `crossing_red_lights` errors describe the car as *correctly stopped*, yet GT says No — the violation is presumably temporal (the car eventually proceeds on red, or approached too fast):

> "The ego car remains stationary at the intersection while the traffic lights for its direction are clearly red throughout the sequence. This behavior is consistent with standard traffic rules" (`cross_red_034_Sunny.mp4`, GT=No, VLM=Yes)

> "The ego car identifies the red traffic light ahead and gradually slows down, coming to a complete stop at the intersection as seen in the final frames" (`cross_red_025_Morning.mp4`, GT=No, VLM=Yes)

**Stop-sign rolling stops mistaken for full stops.** With only 5 frames, "appears to halt" is indistinguishable from a true 3-second stop:

> "It appears to come to a halt or near-halt at the line in the first frame before proceeding to perform a legal left turn" (`cross_stop_018_Night.mp4`, GT=No)

> "The ego car follows the traffic rules by stopping at the stop sign before the intersection" (`cross_stop_018_Morning.mp4`, GT=No)

**Inventing context to dismiss a stop sign.**

> "Although stop signs are visible on the right, they appear to be oriented for the intersecting side streets or adjacent paths, as the lead vehicles and the ego car continue through the intersection without stopping" (`cross_stop_042_Sunny.mp4`, GT=No) — convenient post-hoc rationalisation.

**Fabricating a green light.**

> "approaches an intersection with a clearly visible green traffic light. The vehicle continues to move forward … in accordance with the green signal" (`cross_stop_054_Rainy.mp4`, GT=No) — but the video is in the `crossing_stop` set, so the cue is a stop sign, not a light.

## Hypotheses on Gemini's traffic-law reasoning

1. **It reasons about discrete, frame-visible signals, not motion.** Red lights and stop signs are static, single-pixel-cluster cues that survive 1 fps subsampling. Lane-discipline violations, abrupt lane changes, and tailgating are *trajectory* concepts that 5 frames cannot represent — and Gemini, lacking those frames, defaults to "looks normal".

2. **Strong default-to-compliant prior.** Gemini's positive-class rate is 274/665 = 41% versus a true rate of 10/665 = 1.5%. The bare model has been RLHF-tuned to be cautious about accusing of wrongdoing, so when motion evidence is ambiguous it picks "Yes, compliant". This explains 99% of its errors.

3. **Pattern-recognition style on highway clips, explicit chains on intersections.** Aggressive-takeover evaluations are short, generic, and template-y ("maintains its lane … reasonable speed … no observable violations") — diagnostic of pattern-matching on static highway aesthetics. Intersection evaluations are noticeably longer, cite frame indices, and reference the legal rule (right-on-red, full stop, stop line crossing). The model has a real "intersection rule reasoner" but no equivalent for dynamic lane behaviour.

4. **No realism-vs-rule confusion at intersections.** Gemini is *not* tricked by realistic-looking generated footage when a red light or stop sign is in view — it correctly flags ~71–82% of those cases as violations. The breakdown is purely on subtle dynamics, not on a mistaken belief that "realistic = compliant".

5. **Implication for the agent baseline.** A tool that returns a per-frame trajectory or per-frame lane offset (e.g. RAFT optical flow + lane-line detection) would target exactly the failure mode here. Tools that target FFT/realism (the agentic suite's strength on the Reality category) would not help — Gemini's traffic-law errors are reasoning-from-insufficient-evidence errors, not artifact errors.
