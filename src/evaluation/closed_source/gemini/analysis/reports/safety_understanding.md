# Safety understanding — Gemini 3 Flash Preview

## Safety understanding — summary stats

- Overall accuracy: **1506 / 2122 = 70.97%** (n=2122)
- Two question shapes:
  - Binary "Is it a safe way of driving?" — **779 / 1061 = 73.4%**
  - 1–3 score "1=really dangerous, 2=mostly safe, 3=perfectly safe" — **727 / 1061 = 68.5%**
- Per-folder accuracy makes the picture much sharper:
  - `negative` (regular safe driving in adverse weather): **87.4%**
  - `crossing_red_lights`: **81.6%**
  - `crossing_stop`: **58.9%**
  - `aggressive_takeover`: **19.3%** (catastrophic — 5.7% on the score variant)
- Confusion structure on the binary task is roughly symmetric in count (132 false-unsafe vs 150 false-safe), but the *kind* of error is asymmetric: false-safes pile up almost entirely on `aggressive_takeover` (false-safe rate **67%** there), while false-unsafes pile up on `negative` clips shot in heavy fog/rain/snow.
- On the 1–3 task, 245 of 334 errors (73%) are the **maximally wrong** "GT=1, VLM=3". 191 of those 245 come from `aggressive_takeover` and `crossing_stop`. Symmetric "3→1" errors are vanishingly rare (14 cases total) — the score is biased toward 3.
- Ground-truth label distribution itself is unbalanced: the score variant has 727 GT=3, 485 GT=1, only 31 GT=2. Gemini essentially never produces "2", so it inherits a structural disadvantage on the (rare) middle class.

## What it gets right (with quotes)

Gemini reliably picks up the easy positive cues — lane keeping, following distance, smoothness, absence of erratic maneuvers — and uses them confidently for `negative` clips even under tough weather. It also reads stop signs and red lights from JPEG frames surprisingly well when the signal is clearly visible.

- *Q: Is it a safe way of driving for the ego car? GT/VLM: Yes/Yes (negative, snow):* "The ego car is maintaining a steady position within its lane and keeping a safe following distance from vehicles ahead. Despite the wet and snowy road conditions, there are no signs of erratic driving or immediate hazards in the sequence."
- *Q: 1–3 score. GT/VLM: 3/3 (negative, fog + glare):* "The ego car demonstrates consistent lane keeping and stable movement despite the challenging driving conditions of dense fog and strong sun glare. No hazardous maneuvers, sudden lane departures, or following distance issues are observed."
- *Q: Is it safe? GT/VLM: No/No (crossing_stop):* "The ego car fails to come to a complete stop at the clearly marked 'STOP' sign on the pavement and the vertical stop sign on the right, proceeding straight through the intersection without halting."
- *Q: 1–3 score. GT/VLM: 1/1 (crossing_red_lights, rain):* "The ego car proceeds through the intersection despite a clearly visible red traffic light in the first frame. Driving through a red light, especially in rainy conditions with reduced traction and visibility, constitutes a major safety violation."
- *Q: Is it safe? GT/VLM: No/No (crossing_stop, near-miss):* "The ego car fails to stop or yield at a marked intersection, continuing at speed as a grey car crosses its path from the left. This leads to a near-collision situation in frames 3 and 4."

In short, the model has a working list of textbook safety checks and applies them when the cue is *visually static* (a sign, a light, a lane line). It also correctly downgrades severity in the rare 3→1 cases that involve fog plus tailgating trucks.

## What it gets wrong (with quotes)

The dominant failure mode is **under-calling danger when the cue is dynamic and only visible across frames**. With 5 sampled frames at 1 fps and no real video signal, Gemini cannot estimate ego speed, time-to-collision, lane-change aggressiveness, or whether a "safe gap" was created at the cost of cutting someone off. Every `aggressive_takeover` clip looks like calm highway driving in JPEGs, so Gemini routinely scores it 3.

- *aggressive_takeover, GT/VLM: No/Yes:* "The ego car is driving at a consistent speed on a highway during dusk, maintaining a safe following distance from the vehicles ahead and keeping within its lane. There are no signs of aggressive maneuvers, sudden lane changes, or proximity hazards."
- *aggressive_takeover, GT/VLM: 1/3:* "The ego car maintains a consistent lane position and safe following distance from other vehicles throughout the clip. There are no signs of aggressive driving, speeding, or unsafe maneuvers, making the driving perfectly safe."
- *aggressive_takeover, GT/VLM: 1/3:* "Transitions between lanes for passing are performed smoothly, and a safe distance is maintained from other vehicles under wet road conditions." (the actual clip is the takeover itself — Gemini is describing 5 still snapshots that happen to fall on the calm ends.)
- *crossing_red_lights, GT/VLM: 1/3 (snow):* "It successfully navigates through an intersection as the traffic flow begins, staying centered in its lane and showing no signs of reckless behavior despite the challenging weather." (red light not legible in the sampled frames.)
- *crossing_stop, GT/VLM: 1/3:* "The ego car maintains a safe and steady pace, adhering to its lane and keeping a safe distance from other vehicles and cyclists." (stop sign present in clip but not in the 5 sampled frames.)

The opposite failure — **false-unsafe** — clusters on two sub-cases:
1. **Bad weather priors override behavior**, especially on `negative` fog clips, where Gemini decides high speed in fog is reckless even when the GT label says safe (e.g., GT/VLM: Yes/No: "driving at a relatively high speed through extremely dense fog with very limited visibility").
2. **Sub-clip mislabeling**: a few `negative` and `crossing_stop` videos really do show a red-light or stop violation in the sampled frames; Gemini correctly flags it but the dataset label is `Yes` (GT/VLM: Yes/No: "the overhead traffic light has turned red, which constitutes a major safety violation").

The 1–3 scoring fails worse than the binary version primarily because the scale collapses to 1-vs-3 in Gemini's output (it almost never emits 2). When the visible cues are weakly negative — e.g., a stop-sign rolling stop without cross traffic — binary "No" is still recoverable, but the 3-way version forces it to either commit fully ("1") or revert to its safe-looking-frames default ("3"). On `aggressive_takeover` the score-variant accuracy is **5.7%** vs **32.9%** binary; on `crossing_stop` it is **39.9%** vs **77.8%**. The score format also has only 31 GT=2 clips, and Gemini misses essentially all of them.

## Hypotheses on why Gemini handles safety reasoning well

1. **Safety reasoning is dominated by static, image-recognizable cues.** Lane lines, following distance, stop signs, red lights, weather, and "is the road clear?" are answerable from a single well-chosen frame. With 5 frames, Gemini can usually find at least one that resolves the question. Open-source 7B–8B VLMs trained on image-only data score ~30–40% here because they are noisy on the same cues — Gemini is just a stronger image classifier with stronger reasoning glue.
2. **The class prior is exploitable.** 52% of the dataset is `negative` (label = safe), and Gemini's default narrative ("steady speed, in lane, safe distance") matches it. Even pure prior-following gets ~50%; Gemini then fixes most stop-sign / red-light cases on top of that, which is why the binary score lands at 73%.
3. **Strong language prior for "what makes driving unsafe".** The CoT excerpts read like a checklist transferred from text training data — "stopping distance likely exceeds the visible range", "major traffic violation", "near-collision". When the visual evidence triggers any item on that checklist, Gemini frames the case correctly. This is also where it beats the agentic pipeline: agents pay extra latency and tokens for tool calls that mostly answer "is the lane straight?" — questions Gemini already nails zero-shot.
4. **The remaining 30% of error is precisely the part the agent harness is built for.** `aggressive_takeover` requires *temporal* analysis (relative speed, lane-change velocity); 5 sampled JPEGs systematically destroy that signal. An optical-flow / RAFT tool, or even just dense frame sampling, would close most of this gap. So Gemini's 71% is an upper bound on what *image-only* reasoning can do on this category — and conveniently, the gap to perfect performance maps almost exactly onto motion-cue subsets.
