## Spatial-temporal understanding — summary stats

- **Category accuracy**: 77.6% (263/339).
- **Question mix is dominated by two prompts**:
  - "Has the ego car stopped?" — 184/201 correct (**91.5%**)
  - "On which side is the ego car overtaking?" — 78/133 correct (**58.6%**)
  - Three minor variants ("...after the sign / stop / stop sign") — 1/4 combined.
- **Source split**: `crossing_stop/` 89.8% (185/206), `aggressive_takeover/` 58.6% (78/133). All errors live in these two scenarios.
- **Overtaking confusion** (n=133): GT is "Right" in 127/133 (skewed dataset). Predictions: Right→72 correct, **Left→45**, "It's not overtaking"→10. The 6 GT="Left" cases are 6/6 correct.
- **Stopped confusion** (n=206): 17 false-positives (No→Yes), 4 false-negatives. Gemini leans toward seeing motion when in doubt, except on rainy/night clips with low inter-frame contrast.

## What it gets right (with quotes)

When Gemini succeeds, its reasoning is consistently a chain of *relative-position deltas across the 5 frames* using static landmarks as anchors:

- C2 (cross_stop_030_Rainy, GT=No): "The ego car's position relative to the road markings, stop sign, and crosswalk **changes continuously**, indicating that it does not come to a complete stop."
- C5 (cross_stop_045_Rainy): "The shifting perspective of the roadside scenery and **the motion blur of the road surface** confirm that the vehicle remains in motion." — a rare appeal to blur as a temporal cue.
- C10 (cross_stop_053): "In the first frame, it is behind the stop line, and **by the final frame, it has clearly progressed well into the intersection**." — explicit first-vs-last-frame anchoring.

For correct "Right" overtaking, Gemini anchors on a slower car staying in a fixed adjacent lane while the ego pulls ahead:

- C11 (agg_tk_031): "By frame 4, the ego car has passed the truck, which is clearly positioned to the left of the ego vehicle's path. This indicates the ego car is overtaking the truck on **its right side**."
- C18 (agg_tk_027): "The ego car is seen passing a dark blue vehicle that remains in the lane to its left throughout the sequence. Since the ego car moves past this vehicle while staying to its right, it is overtaking on the right side."

Perception of motion from sparse stills works when (a) a stationary landmark is in frame and (b) the relative-position delta is monotonic.

## What it gets wrong (with quotes)

The dominant failure is **a labeling/convention error on the overtaking question, not a perception error**. In ~40 of the 45 "Left" mistakes, the chain-of-thought localizes the slower car on the ego's right correctly, then labels the maneuver from the *wrong reference frame*:

- W5 (agg_tk_006_Snowy, GT=Right): "The ego vehicle maintains its lane while moving faster than the traffic in **the lane to its right**... Since the ego vehicle is to the left of the cars it passes, it is overtaking **on the left**." — geometry correct, label inverted.
- W11 (agg_tk_017_Rainy): "...it passes a black truck on the right between frames 1 and 2... indicating it is overtaking **on the left**."
- W33 (agg_tk_020_Foggy): "...passing these vehicles from the lane to their left, it is overtaking **on the left side**."

Gemini treats "side of overtaking" = "side of the ego relative to the overtaken vehicle." The benchmark labels it the other way (the lane the ego *uses* to pass). It even exposes the conflict explicitly:

- W0 (agg_tk_002_Foggy): "passing the white sedan on its left (overtaking on the right)... **In standard driving terminology**, the ego car is overtaking the vehicle in the right lane by being on its left side."

A second failure mode is **"It's not overtaking" (10 cases)** when GT is "Right" — usually clips where the ego is being passed, or it passes only one slow car without lane-changing:

- W42 (agg_tk_020_Sunny): "multiple vehicles... pass the ego car on the left, indicating the ego car is... not performing any overtaking maneuvers."
- W48 (agg_tk_009_Golden hour): "Although it passes a slower-moving brown SUV in the adjacent right lane, it does not perform a deliberate **lane change**." — Gemini imposes a stricter definition of overtaking than the benchmark.

For "stopped," the 17 false-positives are clips where the car briefly creeps and Gemini collapses 3 near-identical frames into "stationary":

- W2 (cross_stop_020_Rainy, GT=No): "the position of the car relative to the road markings... **remains identical**, indicating it has stopped." — sub-pixel motion at 1 fps is invisible.
- W29 (cross_stop_054_Rainy): "Throughout frames 1 to 3, the ego car remains in a fixed position... **Movement only begins to be visible in frame 4 and 5**."

Gemini almost never hedges about temporal sparsity — none of the 76 wrong evaluations contain phrases like "limited temporal resolution"; it confidently extrapolates.

## Hypotheses: spatial-temporal reasoning from sparse frames

1. **Static-landmark anchoring works.** Gemini's strategy is "compare object X's pixel position in frame 1 vs frame 5." This is why `crossing_stop` reaches 89.8% — road surface, stop lines, and buildings give a dense reference frame.
2. **The 45-case "Left" cluster is a convention failure, not a vision failure.** Gemini perceives the geometry correctly almost every time; it just maps the geometry to "overtaking on the left" using a *passenger-side reference* instead of the *which-lane-did-the-ego-move-into* benchmark convention. A prompt clarification would likely flip most of these and push the category above 90%.
3. **The right-skewed dataset masks a real bias**: 6/6 GT="Left" cases are correct, so left-vs-right vision is fine — the error is asymmetric only because GT is asymmetric.
4. **1 fps is below the threshold for slow-creep detection.** False-positive stops cluster on rainy / night / snowy clips where contrast is low and inter-frame ego displacement is small. Gemini rarely cites blur or compression as cues; it relies almost entirely on landmark displacement.
5. **No temporal-uncertainty hedging.** Gemini commits to a definite answer in 100% of wrong cases. It doesn't exploit "I'm not sure" even when sparse-frame evidence is genuinely ambiguous (stop-then-go clips), inflating confident wrong answers.
6. **"It's not overtaking" misuse** combines a stricter definition (requires a lane change) with apparent recency bias toward the last frame attended to (W42, W44 latch onto frames where another car passes the ego).

Net: the headline 77.6% understates Gemini's underlying spatial-temporal *perception* — fix the overtaking-side convention and the lead over GPT-5.4-mini in this category would widen substantially.
