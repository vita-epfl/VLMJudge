#!/bin/bash
# Startup for the realgen ablation agent variants (v4o + v4p).
# Args:
#   $1 = absolute path of the questions JSON (with or without hints)
#   $2 = absolute path of the output JSON
#
# Mirrors agentic/startup.sh — only difference is that we pass --questions_file
# and --output_file_path to the patched agent so we can target the 100-video
# realgen subset. The python venv install list is identical.
set -euo pipefail

QUESTIONS_FILE="$1"
OUTPUT_FILE="$2"

# Start vLLM server in background — same flags as agentic/startup.sh
vllm serve /mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/models/Qwen3-VL-30B-A3B-Instruct \
  --tensor-parallel-size 4 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --max-model-len 128000 \
  --async-scheduling \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --mm-processor-kwargs '{"fps": 24.0}' \
  --host 0.0.0.0 \
  --port 8000 &

echo "Waiting for vLLM server to start on port 8000..."
timeout=3600
start_time=$(date +%s)
while ! (echo > /dev/tcp/localhost/8000) >/dev/null 2>&1; do
  current_time=$(date +%s)
  elapsed=$((current_time - start_time))
  if [ -n "$timeout" ] && [ $elapsed -gt $timeout ]; then
    echo "Timeout reached while waiting for vLLM server."
    exit 1
  fi
  sleep 60
done
echo "vLLM server is ready."

# Copy code into the pod so the patched script + agentic/utils/ live together
# (sys.path.insert inside main_multi_tools_multithreaded.py expects
# parents[2]/agentic, which becomes <pod_home>/agentic once both folders sit
# at the pod's CWD).
cp -r /mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/agentic ./
cp -r /mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/realgen_ablation ./
cd realgen_ablation/scripts

# Idempotent dataset build (seed 42 — re-running just overwrites with identical content)
python ../build_realgen_dataset.py \
    --storage_path /mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset \
    --out_dir /mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/realgen_ablation

# venv setup — same package list as agentic/startup.sh
python -m venv ./venv
source ./venv/bin/activate
pip install decord qwen-agent python-dateutil 'imageio[ffmpeg]' av filelock opencv-python kernels soundfile librosa accelerate sentencepiece protobuf
# Transformers strategy: previous canonical install used `git+...main`, which
# now ships a broken CLIPTextModelWithProjection lazy-import that fails when
# Sam3VideoModel is loaded by utils/sam.py. Pin to latest PyPI stable so the
# import chain is consistent; if Sam3VideoModel was added in 5.x stable, this
# works; if not, the failure is a clean ImportError instead of a broken lazy
# loader. Add accelerate/sentencepiece/protobuf for SAM/CLIP runtime deps.
pip install --upgrade transformers
pip3 install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

mkdir -p /mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/realgen_ablation/results
# Switched to v4 (RAFT + SAM + FFT). Requires fft/ package at <pod_home>/fft/ —
# main_multi_tools_v4.py adds parents[2] (= <pod_home>) to sys.path for `from
# fft.compute_fft import process_video_fft`.
cp -r /mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/fft ../../
python main_multi_tools_v4.py \
    --num_workers 15 \
    --questions_file "$QUESTIONS_FILE" \
    --output_file_path "$OUTPUT_FILE"
