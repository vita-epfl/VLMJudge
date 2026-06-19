#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distributed Batched VideoQA Inference for LLaVA-OneVision using vLLM
Tested with: vLLM >= 0.6.0, LLaVA-OneVision-Qwen2-7B/72B-OV
Supports tensor-parallel across multiple GPUs.
"""

import argparse
import json
import os
import traceback
import time
from typing import List, Dict, Any
import logging
from tqdm import tqdm
from decord import VideoReader
import numpy as np

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset  # Only for reference, we use custom loading


# ========================================
# Argument Parser
# ========================================
def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-OneVision VideoQA Batched Inference with vLLM")
    parser.add_argument("--model_path", type=str,
                        help="Path to LLaVA-OneVision model (HF format)",
                        default="/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/models/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument("--dataset_json", type=str, default="/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/dataset/Questions.json",
                        help="Path to Questions.json containing list of message dicts")
    parser.add_argument("--output_file", type=str, default="/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/results/llava_onevision_results.json",
                        help="Output JSON file path")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Effective batch size (adjusted for GPU memory; 32–64 typical on 4x80GB)")
    parser.add_argument("--num_frames", type=int, default=15,
                        help="Number of video frames to sample (15 is recommended for OneVision)")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.93)
    parser.add_argument("--max_model_len", type=int, default=16384)
    return parser.parse_args()


# ========================================
# 3. Video Frame Sampling Utility
# ========================================
def sample_video_frames(video_path: str, num_frames: int = 32) -> np.ndarray:
    """
    Sample num_frames from video using decord.
    Returns: (num_frames, H, W, 3) uint8 numpy array (contiguous, positive strides).
    """
    try:
        vr = VideoReader(video_path)
        total_frames = len(vr)
        if total_frames == 0:
            raise ValueError("Empty video")
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # Shape: (T, H, W, 3), dtype=uint8, BGR
        frames_rgb = frames[..., ::-1].copy()  # BGR → RGB, then .copy() to fix negative strides
        return frames_rgb
    except Exception as e:
        logging.error(f"Failed to process video {video_path}: {e}")
        raise

if __name__ == '__main__': 
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # ========================================
    # Logging
    # ========================================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("llava_onevision_vqa.log")
        ],
    )
    log = logging.getLogger(__name__)

    # ========================================
    # 1. Initialize vLLM LLM with Tensor Parallelism
    # ========================================
    log.info(f"Loading LLaVA-OneVision from {args.model_path} with TP={args.tensor_parallel_size}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,  # Recommended for multimodal models
        max_num_batched_tokens=32768,
        max_num_seqs=args.batch_size,
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    log.info("LLaVA-OneVision loaded successfully across GPUs")

    # ========================================
    # 2. Load Questions + Resume Logic
    # ========================================
    with open(args.dataset_json, "r") as f:
        messages = json.load(f)

    log.info(f"Loaded {len(messages)} questions from {args.dataset_json}")

    # Resume from existing output
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            try:
                existing_outputs = json.load(f)
                existing_outputs = [o for o in existing_outputs if isinstance(o, dict) and "answer" in o]
            except:
                existing_outputs = []
        processed = len(existing_outputs)
        messages = messages[processed:]
        log.info(f"Resuming from {processed} already processed entries")
    else:
        existing_outputs = []
        processed = 0

    outputs = existing_outputs[:]

    


    # ========================================
    # 4. Main Batched Inference Loop
    # ========================================
    total_items = len(messages)
    pbar = tqdm(total=total_items + processed,
                initial=processed,
                desc="VideoQA Progress",
                unit="video",
                colour="green")

    batch_inputs = []
    batch_metadata = []

    for idx, msg in enumerate(messages):
        try:
            # Parse message structure (standard OpenAI-like format)
            content = msg["content"]
            video_item = next(c for c in content if c["type"] == "video")
            text_item = next(c for c in content if c["type"] == "text")

            video_path = video_item["video"]
            question = text_item["text"]

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")

            # Sample frames
            frames_np = sample_video_frames(video_path, num_frames=args.num_frames)

            # Construct correct LLaVA-OneVision prompt
            prompt = f"<|im_start|>user\n<video>\n{question}<|im_end|>\n<|im_start|>assistant\n"

            # Build vLLM input with multi_modal_data
            request_input = {
                "prompt": prompt,
                "multi_modal_data": {
                    "video": frames_np  # Shape: (num_frames, H, W, 3), uint8, RGB
                }
            }

            metadata = {
                "video_path": video_path,
                "question": question,
                "index": processed + idx
            }

            batch_inputs.append(request_input)
            batch_metadata.append(metadata)

        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Failed to process video index {processed + idx} | Path: {video_path if 'video_path' in locals() else 'UNKNOWN'} | Error: {str(e)}"

            log.error(error_msg)                     # ← visible in console + log file
            log.debug(tb)  # optional: full traceback only in debug mode
            
            pbar.update(1)
            continue

        # Process full batch
        if len(batch_inputs) == args.batch_size or (idx == len(messages) - 1):
            if not batch_inputs:
                continue

            log.info(f"Generating batch of {len(batch_inputs)} videos...")

            # Start timing the inference
            inference_start_time = time.time()
            vllm_outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
            # End timing the inference
            inference_end_time = time.time()
            batch_inference_time = inference_end_time - inference_start_time

            for vllm_out, meta in zip(vllm_outputs, batch_metadata):
                generated_text = vllm_out.outputs[0].text.strip()

                result = {
                    "video": meta["video_path"],
                    "question": meta["question"],
                    "answer": generated_text,
                    "inference_time": batch_inference_time / len(batch_inputs)
                }

                outputs.append(result)

            # Update progress
            pbar.update(len(batch_inputs))

            # Save intermediate results
            with open(args.output_file, "w") as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)

            log.info(f"Saved {len(outputs)} results so far")

            # Clear batch
            batch_inputs.clear()
            batch_metadata.clear()

    # Final save
    with open(args.output_file, "w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    pbar.close()
    log.info("Inference completed!")
    print(f"\nAll done! Results saved to: {args.output_file}")
    print(f"Total processed: {len(outputs)}")