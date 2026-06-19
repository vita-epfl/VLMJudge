# Reality understanding — Gemini 3 Flash Preview

## Reality understanding — summary stats

- **Overall accuracy:** 51.10% (1597 / 3125) — the lowest of all categories and the only one where Gemini sits well below its 64.0% global average.
- **Question split:**
  - Binary `Real vs Generated`: 870 / 1571 = **55.4%**.
  - Realism score 1–3: 726 / 1553 = **46.7%**.
- **Ground-truth distribution in this slice:** every binary item has GT = `Generated`; every score item has GT ∈ {1, 2}. There are zero real clips in the eval, so the binary task collapses to "spot the artifact". The model nevertheless outputs `Real` 701 times — that is the entire binary error mass.
- **Score confusion (wrong cases only):**
  - GT=2 → VLM=3: **359** (over-rate "minor issues" as photoreal)
  - GT=1 → VLM=2: **277** (down-rate "clearly impossible" to "mostly realistic")
  - GT=1 → VLM=3: **100** (extreme miss)
  - GT=2 → VLM=1: **91** (over-flag minor-issue scenes as impossible)
- **Per-source accuracy on the binary task:**
  - cosmos-drive-dreams: **65.8%** (669/1017)
  - cosmos-predict1 (`negative/…`): **36.3%** (201/554) — Gemini essentially fails the cosmos-predict1 reality test.
- **Per-source on the score task:** drive-dreams 47.0% vs predict1 46.4% — equally hard.
- **Per-GT score accuracy:** GT=1 → 38.1%, GT=2 → 52.3% — the model systematically pulls scores toward the middle (mode collapse around "2/3").

## What it gets right (with quotes)

When Gemini is right on this category, it is almost always because it spotted a hard, named artifact in the 5 sampled frames — most often **garbled text, morphing objects, or violated geometry**. Nonsensical signage is the single most-cited cue.

- *(Generated → Generated, drive-dreams)* `crossing_stop/cross_stop_012_Morning.mp4` — "the morphing of the **'STOPP'** text on the road and the shifting, inconsistent appearance of the stop signs and tree foliage across frames. The textures of the trees are particularly **'mushy'** and lack the structural integrity of real-world objects."
- *(Generated → Generated, predict1)* `negative/3d656470…_Snowy.mp4` — "the nonsensical text painted on the road surface, which morphs into **'STNULL - STNAID'** as the camera approaches. … the signage on the street lamp pole is composed of garbled, unreadable characters."
- *(Generated → Generated, drive-dreams)* `artifacts/car_brake_008_Rainy.mp4` — "the 'PARKING' sign in the background morphing into **'PANNI'** and the orange 'Flywheel' taxi on the right disappearing instantly between the first and second frames."
- *(GT=1 → 1, drive-dreams)* `artifacts/car_crash_018_Snowy.mp4` — "garbled text on license plates, and **nonsensical traffic signals including heart shapes and conflicting lights**. These major logical and structural inconsistencies make the scene clearly impossible."
- *(GT=1 → 1, drive-dreams)* `aggressive_takeover/agg_tk_009_Night.mp4` — "the **moon** that incorrectly appears as a **ring or donut shape**, which is physically impossible."
- *(Generated → Generated, predict1)* `negative/e0dd616e…_Original.mp4` — "The lens flare in the upper-right section is physically implausible, appearing as a **large, solid, curved glowing mass** that does not follow standard optical patterns."

The recurrent winning cues are: (1) misspelled / morphing text on signs, lane markings, license plates; (2) objects that pop in/out or change identity across the 5 frames (cars dissolving, "vanishes between frame 2 and 3"); (3) impossible optics (donut moons, solid lens flares); (4) "mushy" / painterly foliage.

## What it gets wrong (with quotes)

The dominant failure pattern is **over-trusting visually clean cosmos-predict1 clips** (the high-fidelity model). When 5 stills look photographic, Gemini confidently produces a "Real" verdict even though the file is generated.

- *(GT=Generated → Real, predict1)* `negative/e255e893…_Foggy.mp4` — "Object consistency, including the palm trees, streetlights, and passing vehicles, remains stable throughout the sequence without any signs of AI-generated warping or artifacts."
- *(GT=Generated → Real, predict1)* `negative/08bc010a…_Morning.mp4` — "perfectly stable road markings (**'ONLY'**) and building signage as the camera moves. There are no signs of AI-generated artifacts such as flickering, morphing objects, or illogical physical movements, indicating it is a **real dashcam recording**."
- *(GT=Generated → Real, drive-dreams)* `aggressive_takeover/agg_tk_006_Sunny.mp4` — "Real-world details such as the **'Mollie Stone's' grocery store** sign and 'Tuttle's' pharmacy building are accurately depicted without any morphing or AI-typical hallucinations." (Gemini treats *plausible-sounding* business names as evidence of authenticity, but the entire scene is synthetic.)
- *(GT=1 → 3, drive-dreams)* `artifacts/Xy_FYVEFVnk_28704_14b.mp4` — "legible brand signage like **'PALCONERI' and 'ROLEX'**. The movement of the black SUV and pedestrians follows logical paths without any warping or common AI generation artifacts." ("PALCONERI" is itself a hallucinated brand — the model treats the readable garbage as real.)
- *(GT=1 → 2, drive-dreams)* `artifacts/pedestrians_012_Morning.mp4` — "pedestrian movement reveals AI artifacts. The pedestrians appear to **slide rather than walk**… frame 5 shows unnatural limb morphing." Despite naming a structural failure, Gemini still rates the scene "mostly realistic with minor issues" rather than "clearly impossible".
- *(GT=2 → 1, drive-dreams)* `aggressive_takeover/agg_tk_029_Sunny.mp4` — calls "garbled text … blue sign floating in the sky … sun rendered as an unnatural oversized white blob" but votes 1 when the GT is 2; an over-correction in the opposite direction.

Two systematic biases are visible:
1. **Inter-frame independence.** With only 5 sampled frames at 1 fps, "temporal" arguments are mostly imagined. When nothing obvious moves between five stills, Gemini concludes "high temporal consistency" and flips to Real — see the predict1 quotes above.
2. **Score regression to the mean.** Even when Gemini correctly enumerates impossibilities (sliding pedestrians, melted trucks, "STTOP"), it grades them 2 instead of 1, and conversely upgrades clean GT-2 scenes to 3. The 1↔3 confusions (100 cases) show the most extreme manifestation — the model genuinely believes some clearly-synthetic clips are indistinguishable from real.
3. **Source asymmetry.** cosmos-predict1 binary accuracy is 36.3% vs 65.8% for cosmos-drive-dreams — predict1 outputs are visually closer to natural dashcam footage and Gemini falls for them with high confidence ("indistinguishable from real dashcam footage" appears verbatim in dozens of wrong evaluations).

## Hypotheses on why Gemini outperforms baseline VLMs in this category

Even at 51%, Gemini's reality score is well above the open-source baselines, and its win pattern points to three things:

1. **Strong OCR + world-knowledge on signage.** The single most reliable cue across correct cases is detecting that text is *almost* English ("STOPP", "STNULL", "PANNI", "CLLVER", "PALCONERI"). This requires both legible-character extraction from low-res frames *and* a language model that flags the strings as not-real-words. Most open VLMs that score < 30% here are weaker on either side of that pipeline — they read coarse text and rarely cross-check it against world knowledge.
2. **Concrete artifact taxonomy in chain-of-thought.** Gemini's reasoning consistently invokes a small, named vocabulary — "morphing", "temporal flickering", "physically implausible lens flare", "mushy textures", "structural integrity" — that maps directly onto the artifacts cosmos models produce. This looks like a learned prior from generative-model evaluation data, not generic image description. Open VLMs tend to describe the scene rather than diagnose it.
3. **Calibration on the 5-frame budget.** The agentic pipeline (60.1%) wins partly because tools like FFT and optical flow exploit *all frames*; Gemini, with only 5 stills and no audio, must rely on per-frame cues. It still nearly matches the agent on the binary task because per-frame artifacts (text, optics, geometry) carry most of the signal — but it loses the score task and predict1 clips because *temporal* artifacts are exactly what 1 fps subsampling destroys. The same scaffold (Gemini-3-Flash) on full-video input would likely close most of the remaining gap to the agent.

In short: Gemini is a strong **per-frame artifact spotter with a generative-model-aware prior**, and that prior alone is enough to beat baselines that lack it; what holds it back is sparse temporal sampling and a tendency to score by gestalt impression rather than by a strict "any artifact ⇒ rating 1" rule.
