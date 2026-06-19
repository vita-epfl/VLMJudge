# Artifacts understanding — Gemini 3 Flash Preview

## Summary stats

- Category accuracy: **83.9%** (610 / 727), Gemini's 2nd-best category, n=727.
- Two questions split the category roughly 55 / 45:
  - "Do some objects change shape or appearance ?" — n=402, acc **92.0%** (377/402).
  - "Does any object appear or disappear without continuous motion ?" — n=325, acc **76.0%** (247/325).
- Heavy class imbalance: GT=Yes 643, GT=No 84 (88% positive). Gemini predicts Yes 672 / 727 (92.4%) — slightly more biased toward Yes than the data itself.
- Confusion (combined): TP=599, TN=11, FP=73, FN=44. Errors split roughly 62% false positives, 38% false negatives.
- Generator-tag breakdown shows the FN cluster is almost entirely from clips ending in `_7b.mp4` (24 of 44 FN), versus only 3 FN on `_14b.mp4`. The smaller cosmos generator produces more subtle artifacts, which Gemini under-detects: per-question accuracy on `_7b.mp4` clips is **76.6%** (shape) and **67.9%** (appear/disappear), versus **95.9%** and **78.2%** on `_14b.mp4`.

## What it gets right (with quotes)

When artifacts are visually loud, Gemini latches on quickly and writes very specific frame-localised explanations. Consistent failure modes it spots:

1. **Garbled signage / on-road text** — by far the most-cited cue. Stop-text mangling and license-plate flicker are flagged in dozens of correct calls:
   - "the 'STOP' text on the road (mangled as 'ITO') and the stop sign on the right side of the street near the bus disappear abruptly between the first and subsequent frames" (`cross_stop_016_Rainy.mp4`).
   - "the license plate and rear lights of the silver car on the right fluctuate in appearance between frames 4 and 5" (`uvRaCzWpFrQ_69264_14b.mp4`).
   - "the 'STOP' sign on the left side of the street warps and eventually disappears, while the yellow diamond sign above it changes its internal markings" (`cross_stop_010_Golden hour.mp4`).

2. **Vehicles disappearing mid-frame / morphing identity** — distinct from "drove out of view":
   - "The silver sedan directly in front of the black SUV in the first frame disappears instantly in the second frame" (`cross_stop_010_Snowy.mp4`).
   - "the white van in the center lane suddenly morphs into a police car and pedestrians" (`pedestrians_010.mp4`).
   - "the silver car approaching in the oncoming lane … changes from a small sedan/hatchback into a larger SUV with distinct black wheels" (`car_crash_017.mp4`).

3. **Body-shape distortion of pedestrians and wheels**:
   - "the person in the brown coat … becomes unnaturally thin and elongated, eventually appearing to float … with distorted limbs" (`pedestrians_018_Rainy.mp4`).
   - "its front-left wheel turns into a solid black disc" (`cross_stop_009.mp4`).

4. **Localised flicker / appearance-only artifacts** — the windshield wiper morph in `cross_red_019_Foggy.mp4` ("multiple morphing black limbs that do not move in a continuous or realistic manner"), pink light streaks, etc.

In the 247 correct "appear/disappear" answers, 35% of evaluations explicitly localise to a frame index (`frame 3`, `between frames 4 and 5`), 37% use the phrase "continuous motion" verbatim, and 30% explicitly attribute the cause to AI generation. Gemini is clearly *reasoning over discrete pairs of frames*, not over a perceived video.

## What it gets wrong (with quotes)

### False positives (73 of 117 errors): over-eager artifact detection

Gemini calls "Yes" on 73 real-video clips. The same vocabulary it uses on AI-gen clips bleeds onto realistic phenomena:

- **Genuine occlusion mistaken for popping** — "A person wearing a large dark jacket … In frame 5, this person completely disappears from the scene without any visible motion" (`pedestrians_008_Foggy.mp4`, real). At 1 fps a pedestrian can easily be occluded by a passing car between samples; Gemini cannot distinguish.
- **Real text genuinely changing across frames** (different signs / different vehicles) — "the digits on a speed limit sign in the background change inconsistently (from 25 to 56, 33, and then 23)" (`cross_stop_013.mp4`, real). These are different signs the camera is passing.
- **Camera-pan or rearview-mirror entering frame** — "a rearview mirror suddenly appears at the top of the frame in Frame 4" (`cross_red_019_Rainy.mp4`, real). Hood/mirror occlusion shifts as the dashcam adjusts.
- **License-plate text "morphing"** — Gemini reliably reads license plates as morphing across nearly every clip (`car_crash_009.mp4`, `car_brake_002_Morning.mp4`, `car_brake_005_Sunny.mp4`, `cross_red_017.mp4`). Real license plates legitimately blur differently across frames at 1 fps with motion blur and rolling shutter, which Gemini mis-classifies.
- **Cars exiting the lane / lane changes** — "a black sedan visible between an SUV and a red car in the first frame disappears and is replaced by a different dark vehicle" (`JS0gJxhFFJ8_7344_14b.mp4`, real). Real traffic at 1 fps can move several car-lengths, so a car *can* legitimately leave a slot.

### False negatives (44 of 117): subtle artifacts in `_7b` clips

Gemini misses softer artifacts almost entirely on the smaller-generator cosmos clips. The chains-of-thought are notable for what they *don't* contain — zero of the 44 FN evaluations cite a specific frame index (vs ~35% in correct calls). They are generic "everything looks fine" boilerplate:

- "all objects … move consistently with the camera's forward motion. While some objects enter or exit the frame between samples, this is due to the 1 frame-per-second sampling rate" (`Qb5UT3pSxjI_79968_7b.mp4`).
- "While there are some AI-related morphing artifacts, such as the characters on the license plate changing, no objects suddenly appear or disappear without continuous motion" (`lT4AML9VU2Q_67344_7b.mp4`) — Gemini sees morphing but compartmentalises it as not-relevant to the question.
- "all objects … maintain a consistent shape and appearance" (`car_brake_009_Sunny.mp4`, `cross_stop_017.mp4`, etc.).

In two FN cases Gemini explicitly invokes the 1 fps sampling rate as an exculpatory explanation — i.e. it *is* aware of the sparse sampling and uses it as a default "benefit of the doubt" prior, but applies it inconsistently (it does not extend the same charity in the FP cases).

## Hypotheses: how does Gemini detect temporal artifacts from 5 still frames?

The chains-of-thought make Gemini's strategy fairly transparent:

1. **It does pairwise frame-to-frame consistency checks.** Phrases like "between frame 1 and frame 2", "in the transition from frame 4 to frame 5" appear in 35% of correct appear/disappear calls and in 21% of FPs. It is treating the input as five labelled images and asking "does object X in frame i have a plausible counterpart in frame i+1?". The model never uses the word "video" in its reasoning the way a true video model would.

2. **It uses scene-logic / object-permanence priors, not optical flow.** When a stop sign on the right side of the road disappears between adjacent frames while the camera is still pointed at the same intersection, Gemini flags it because *the camera geometry implies the sign should still be visible*, not because it computed motion. This is why it scores well: object-permanence reasoning over 5 frames generalises remarkably well to "without continuous motion" — Gemini reformulates the question as "does object X have a logically consistent location in the next frame I can see?".

3. **It leans heavily on text-rendering failures.** Garbled stop-text, mangled license plates, and morphing storefront names are by far the most cited cue. Gemini is essentially using *static* per-frame OCR-like inconsistencies as a strong proxy for temporal AI artifacts. This is robust on cosmos-14b (which produces obvious gibberish text) and weak on cosmos-7b (which sometimes produces sharper text or where the artifacts are confined to vehicle silhouettes the model deems acceptable).

4. **Yes-bias is the dominant error mode.** With 88% of GT=Yes the prior is heavy, and Gemini is happy to call any plausible-looking inconsistency an artifact. It mostly fails the small no-artifact subset (10 / 84 GT=No correct), which is consistent with a model that was prompted to look for artifacts and almost always finds something to flag.

5. **The 1 fps sampling rate is a known confound to the model itself.** It surfaces the phrase "1 fps / sampling rate" only in FN reasoning (2 of 44), suggesting that when Gemini consciously remembers the sparse sampling it tends to over-correct toward "No". This is the lever an agent harness or a refined prompt could plausibly exploit to recover the missed `_7b` cases without inflating false positives.
