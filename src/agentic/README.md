# 🤖 src/agentic/

A `qwen-agent` pipeline that wraps **Qwen3-VL** with perception tools and lets it *investigate* a
clip before answering. It talks to a **local vLLM server that must already be running on port 8000**
(`cluster/startup_v4.sh`, kept locally, starts the server, waits for the port, then runs the
agent). Run it with the agent dir as the working directory:
`cd src/agentic && python main_multi_tools_v4.py`.

**Contents**
- `main_multi_tools_v4.py` — the winning agent (`v4o` / `v4p`). Registers `get_motion_info` (RAFT),
  `get_masks` (SAM), `get_frequency_analysis` (FFT) and `final_answer` (`{evaluation, answer}`).
- `main_multi_tools_timing.py` — latency-instrumented variant (feeds the timing study).
- `utils/raft.py`, `utils/sam.py` — the optical-flow and segmentation tool workers.
- `fft/compute_fft.py` — the FFT spectral tool (the cue that exposes generation artifacts).
- `postprocess_missing.py` — recovers answers from raw output when `final_answer` wasn't called.

This dir is self-contained: every import (`utils.*`, `fft.compute_fft`) resolves with `src/agentic/`
as the working directory. See the [root README](../../README.md).
