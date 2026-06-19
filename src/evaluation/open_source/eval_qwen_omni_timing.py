#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import traceback
import time
from tqdm import tqdm

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info


# ========================================
# Argument Parser
# ========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Batched inference with Qwen3-Omni-30B-A3B-Instruct + metadata")

    parser.add_argument("--storage_path", type=str, default="/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_file_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8, help="Recommended ≤8 for 30B model")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--use_audio_in_video", action="store_true", default=False)
    parser.add_argument("--no_audio_in_video", dest="use_audio_in_video", action="store_false")
    parser.add_argument("--save_with_metadata", action="store_true", default=True,
                        help="Save video path + question + answer (default: True)")
    parser.add_argument("--no_metadata", dest="save_with_metadata", action="store_false",
                        help="Save only the raw answer text")

    return parser.parse_args()


args = parse_args()

# Resolve paths
if args.model_path is None:
    args.model_path = os.path.join(args.storage_path, "models", "Qwen3-Omni-30B-A3B-Instruct")
if args.output_file_path is None:
    args.output_file_path = os.path.join(args.storage_path, "results", "Qwen_omni_results.json")

os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)

# ========================================
# Logging
# ========================================
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(args.storage_path, "qwen_omni_batch.log"))
    ],
)
log = logging.getLogger(__name__)

# ========================================
# Load Model & Processor
# ========================================
log.info("Loading model from %s", args.model_path)
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    args.model_path,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.disable_talker()
processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)
log.info("Model and processor loaded")

# ========================================
# Load Questions & Resume Logic
# ========================================
questions_path = os.path.join(args.storage_path, "dataset", "Questions.json")
with open(questions_path) as f:
    all_messages = json.load(f)

log.info("Loaded %d questions from %s", len(all_messages), questions_path)

# Resume from existing output file
if os.path.exists(args.output_file_path):
    with open(args.output_file_path) as f:
        outputs = json.load(f)
    # Keep only successful dicts that contain "answer"
    outputs = [o for o in outputs if isinstance(o, dict) and "answer" in o]
else:
    outputs = []

already_processed = len(outputs)
messages = all_messages[already_processed:]
log.info("Resuming – %d already processed, %d remaining", already_processed, len(messages))

# ========================================
# Batched Inference
# ========================================
batch_outputs = []

for batch_idx, start_idx in enumerate(tqdm(range(0, len(messages), args.batch_size), desc="Batches")):
    batch_messages = [[msg] for msg in messages[start_idx:start_idx + args.batch_size]]
    log.info("=== Batch %d | %d samples ===", batch_idx, len(batch_messages))

    # ------------------------------------------------------------------
    # 1. Process each batch
    # ------------------------------------------------------------------

    try :
        # Preparation for inference
        text = processor.apply_chat_template(
            batch_messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        audios, images, videos = process_mm_info(batch_messages, use_audio_in_video=args.use_audio_in_video)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=args.use_audio_in_video,
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output text and audio
        # Start timing the inference
        inference_start_time = time.time()
        text_ids, _ = model.generate(
            **inputs,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=args.use_audio_in_video,
            max_new_tokens=args.max_new_tokens,
            return_audio=False,
        )
        # End timing the inference
        inference_end_time = time.time()
        batch_inference_time = inference_end_time - inference_start_time

        # Trim prompt tokens using attention_mask sums (per-sample prompt lengths)
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        trimmed_ids = [seq[p_len:] for seq, p_len in zip(text_ids.sequences, prompt_lengths)]

        answers = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for message, txt in zip(batch_messages, answers):
            if args.save_with_metadata:
                path = message[0]['content'][0]['video']
                q = message[0]['content'][1]['text']
                batch_outputs.append({
                    "video": path,
                    "question": q,
                    "answer": txt,
                    "inference_time": batch_inference_time / len(batch_messages)
                })
            else:
                batch_outputs.append(txt)

    except Exception as e:
        log.error("Sample processing failed: %s", str(e))
        # On full batch failure, add error entries for all samples
        for msg in batch_messages:
            video_path = "UNKNOWN"
            question = ""
            for item in msg["content"]:
                if item["type"] == "video":
                    video_path = item["video"]
                if item["type"] == "text":
                    question = item["text"]
            error_entry = {
                "video": video_path,
                "question": question,
                "answer": f"[BATCH ERROR] {str(e)}"
            }
            batch_outputs.append(error_entry if args.save_with_metadata else error_entry["answer"])

    # ---------- Save intermediate results ----------
    outputs.extend(batch_outputs)
    with open(args.output_file_path, "w") as f:
        json.dump(outputs, f, indent=4)
    log.info(" → batch saved – %d new entries", len(batch_outputs))

    # ---------- Cleanup ----------
    torch.cuda.empty_cache()
    batch_outputs=[]

print("Inference completed")
print(f"Results with metadata saved to: {args.output_file_path}")
print(f"Total entries: {len(outputs)}")