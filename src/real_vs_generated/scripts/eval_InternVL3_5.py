#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import traceback
from tqdm import tqdm

import torch
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import logging

from lmdeploy import pipeline, PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.vl.constants import IMAGE_TOKEN


# ========================================
# Argument Parser
# ========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Batched inference with InternVL3.5-8B via LMDeploy")

    parser.add_argument(
        "--storage_path",
        type=str,
        default="/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval",
        help="Root directory containing models/, dataset/, and where logs/results will be stored",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default=None,
        help="Full path to the output JSON file. If not provided, will be <storage_path>/results/InternVL3_5_results.json",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Number of videos to process in each batch (default: 20)",
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=15,
        help="Target number of frames to subsample from the video (default: 8)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate per answer (default: 1024)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=2,
        help="Tensor parallelism - number of GPUs",
    )
    parser.add_argument(
        "--session_len",
        type=int,
        default=32768,
        help="Session len of the model",
    )
    parser.add_argument(
        "--cache_max_entry_count",
        type=float,
        default=0.8,
        help="optional: limits KV cache occupancy",
    )
    parser.add_argument(
        "--pytorch_backend",
        type=bool,
        default=False,
        help="optional: limits KV cache occupancy",
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default="dataset/Questions.json",
        help="JSON file containing the list of questions (same format as Aria benchmark)")
    # SURGICAL PATCH: allow overriding the hardcoded InternVL3.5 model dir
    # so a single script can run both the 30B and 8B variants.
    parser.add_argument(
        "--model_subdir",
        type=str,
        default="InternVL3_5-30B-A3B-Instruct",
        help="Folder name under <storage_path>/models/ for the InternVL3.5 model to load")

    return parser.parse_args()


# ========================================
# Helper functions (Modified)
# ========================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Removed build_transform and dynamic_preprocess as LMDeploy handles them internally now.

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

# 🚨 MODIFIED: Returns a list of PIL Images, not Tensors.
def load_video(video_path, bound=None, num_segments=32):
    # Note: input_size and max_num are not needed here anymore as LMDeploy handles preprocessing
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    frame_list = []
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    for frame_index in frame_indices:
        # Get frame as numpy array and convert to PIL Image
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        frame_list.append(img)
        
    return frame_list # List of PIL Images

# ========================================
# Main Execution Guard
# ========================================
if __name__ == '__main__':

    args = parse_args()


    # Resolve output_file_path if not explicitly given
    if args.output_file_path is None:
        args.output_file_path = os.path.join(args.storage_path, "results", "InternVL3_5-30B_results.json")

    # Ensure required directories exist
    os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)
    os.makedirs(os.path.join(args.storage_path, "results"), exist_ok=True)

    # ========================================
    # Logging setup (uses storage_path)
    # ========================================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.storage_path, "internvl_batch_debug.log")),
        ],
    )
    log = logging.getLogger(__name__)

    # ========================================
    # Load Model & Processor
    # ========================================
    print("------ Model Loading ------")
    model = f"{args.storage_path}/models/{args.model_subdir}"
    # Using PytorchEngineConfig with tp=2 for 2-GPU Tensor Parallelism
    if args.pytorch_backend:
        log.info("Using Pytorch backend")
        pipe = pipeline(model, 
                        backend_config=PytorchEngineConfig(
                            session_len=args.session_len, 
                            tp=args.tp,
                            cache_max_entry_count=args.cache_max_entry_count
                        ))
        
    else :
        log.info("Using Turbomind backend")
        pipe = pipeline(model,
                        backend_config=TurbomindEngineConfig(
                            session_len=args.session_len,
                            max_batch_size=16,
                            tp=args.tp,
                            cache_max_entry_count=args.cache_max_entry_count,
                            rope_scaling_factor=2.0,  # InternVL3.5 needs this
                        ))
    print("------ Model Loaded ------")

    # ========================================
    # Load Messages and Resume Logic
    # ========================================
    # SURGICAL PATCH: allow absolute --questions_file (for realgen ablation)
    messages_path = args.questions_file if os.path.isabs(args.questions_file) \
        else os.path.join(args.storage_path, args.questions_file)
    with open(messages_path) as m:
        messages = json.load(m)
    print(f"------ {len(messages)} Questions Loaded ------")

    # Resume from existing outputs
    if os.path.exists(args.output_file_path):
        with open(args.output_file_path) as f:
            outputs = json.load(f)
        # Keep only real answers (dicts with "answer" key)
        outputs = [o for o in outputs if isinstance(o, dict) and "answer" in o]
    else:
        outputs = []

    processed = len(outputs)
    
    # 💥 CRITICAL FIX: Ensure we don't try to resume past the end of the dataset
    total_messages = len(messages)
    if processed > total_messages:
        log.warning(f"Resume count ({processed}) is greater than total messages ({total_messages}). Resetting resume count to 0.")
        processed = 0
        outputs = [] # Clear outputs if we are reprocessing the full dataset

    messages = messages[processed:]  # skip already processed entries
    log.info(f"Resuming from {processed} already processed entries. {len(messages)} messages remaining.")

    # ========================================
    # Batched Inference
    # ========================================
    print("------ Running Batched Inference ------")
    
    # Check if there's anything left to process
    if not messages:
        print("------ Nothing to process. Inference Done ------")
        exit()
        
    batch_outputs = []

    for batch_idx, batch_start in enumerate(tqdm(range(0, len(messages), args.batch_size), desc="Batches")):
        batch_messages = messages[batch_start:batch_start + args.batch_size]
        current_batch_size = len(batch_messages)
        log.info("=== Batch %d/%d – %d messages ===",
                 batch_idx, (len(messages)-1)//args.batch_size, current_batch_size)

        batch_video_paths = []
        batch_questions_with_images = [] # Store tuple of (text, list_of_images)
        
        # Tracks which messages successfully generated frames
        successful_message_indices = [] 

        # ---------- 1. Load & subsample videos (as PIL Images) ----------
        for local_idx, msg in enumerate(batch_messages):
            try:
                content = msg["content"]
                video_item = next(c for c in content if c["type"] == "video")
                text_item = next(c for c in content if c["type"] == "text")
                video_path = video_item["video"]
                base_question = text_item["text"]

                log.info(" → video %d/%d: %s", local_idx+1, current_batch_size, video_path)

                if not os.path.exists(video_path):
                    raise FileNotFoundError(video_path)

                # 🚨 MODIFIED: Call new load_video
                frame_list = load_video(video_path, num_segments=args.num_segments)
                
                # Build the InternVL-specific multi-frame prompt format (text part)
                video_prefix = ''.join([f'Frame{i+1}: {IMAGE_TOKEN}\n' for i in range(len(frame_list))])
                full_question = video_prefix + base_question
                
                # LMDeploy expects a list of (text, image/list_of_images) tuples.
                # Here, we combine the text and the list of PIL Images.
                prompt_input = (full_question, frame_list) 

                batch_video_paths.append(video_path)
                batch_questions_with_images.append(prompt_input)
                successful_message_indices.append(local_idx) # Record success

            except Exception as e:
                err = {
                    "video": video_path if 'video_path' in locals() else "UNKNOWN",
                    "question": base_question if 'base_question' in locals() else "",
                    "error": traceback.format_exc()
                }
                log.error(" video %d failed: %s", local_idx+1, str(e))
                #batch_outputs.append(err) # Store failure/error message for saving
                continue

        # ---------- If no video succeeded in the batch ----------
        if not batch_questions_with_images:
            log.warning("All videos in batch failed → skipping generation")
            outputs.extend(batch_outputs) # Save the recorded errors
            with open(args.output_file_path, "w") as f:
                json.dump(outputs, f, indent=4)
            batch_outputs = []
            continue

        # ---------- 2. Build user messages (Prompts for LMDeploy) ----------
        # The prompts are now ready, stored in batch_questions_with_images
        prompts = batch_questions_with_images
        
        log.info("Starting LMDeploy inference for batch with %d requests.", len(prompts)) 

        # ---------- 3. Generate ----------
        answers_response_objects = pipe(prompts, max_new_tokens=args.max_new_tokens)

        log.info("LMDeploy inference completed.")
        
        # 🚨 FIX: Convert LMDeploy Response objects to plain text strings 
        # so they can be JSON serialized.
        answers = [res.text for res in answers_response_objects] 
        
        # The user wanted to see the output, let's print the first few answers:
        #if answers:
        #    print("\n--- Sample Answers (First 3) ---")
        #    for i, ans in enumerate(answers[:3]):
        #        print(f"Answer {i+1}: {ans}")
        #    print("--------------------------------\n")
        
        # ---------- 4. Pair with metadata and Save Results ----------
        
        num_successful = len(answers)
        
        # Extract the original paths and questions corresponding to the successful prompts
        successful_paths = [batch_video_paths[i] for i in successful_message_indices]
        # We need the original, non-prefixed question text for the metadata if desired
        successful_original_questions = [batch_messages[i]["content"][-1]["text"] 
                                         for i in successful_message_indices] 

        for path, q_orig, txt in zip(successful_paths, successful_original_questions, answers):
            outputs.append({
                "video": path,
                "question": q_orig, # Save the clean question text
                "answer": txt
            })


        # ---------- 5. Save intermediate results ----------
        with open(args.output_file_path, "w") as f:
            json.dump(outputs, f, indent=4)
        log.info(" → batch saved – %d successful entries, %d error entries.", 
                 num_successful, current_batch_size - num_successful)

        # ---------- 6. Cleanup ----------
        # No heavy CUDA tensors to clean up from frame loading, but clear lists
        del prompts, answers, batch_questions_with_images
        torch.cuda.empty_cache() # Still good practice
        batch_outputs = []

    print("------ Inference Done ------")
    print(f"Results saved to {args.output_file_path}")