#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import logging
import traceback
from tqdm import tqdm

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


# ========================================
# Argument Parser
# ========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Batched vLLM inference for Cosmos-Reason1-7B with metadata")

    parser.add_argument("--storage_path", type=str,
                        default="/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval",
                        help="Root directory (models/, dataset/, results/)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override model path (default: <storage_path>/models/Cosmos-Reason1-7B)")
    parser.add_argument("--output_file_path", type=str, default=None,
                        help="Output JSON path (default: <storage_path>/results/Cosmos_Reason_results.json)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="vLLM batch size – can be high (16–64) since it's efficient (default: 16)")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Usually 1 unless using multi-GPU")
    parser.add_argument("--save_with_metadata", action="store_true", default=True,
                        help="Save video path + question + answer (default: True)")
    parser.add_argument("--no_metadata", dest="save_with_metadata", action="store_false",
                        help="Save only raw text")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # Resolve paths
    if args.model_path is None:
        args.model_path = os.path.join(args.storage_path, "models", "Cosmos-Reason1-7B")
    if args.output_file_path is None:
        args.output_file_path = os.path.join(args.storage_path, "results", "Cosmos_Reason_results.json")

    os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)

    # ========================================
    # Logging
    # ========================================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.storage_path, "cosmos_reason_vllm.log"))
        ]
    )
    log = logging.getLogger(__name__)

    log.info("Starting Cosmos-Reason1-7B batched inference")
    log.info("Model path: %s", args.model_path)
    log.info("Output path: %s", args.output_file_path)

    # ========================================
    # Initialize vLLM & Processor
    # ========================================
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    log.info("Initializing LLM (tensor_parallel_size=%d)", args.tensor_parallel_size)
    llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"image": 0, "video": 1},
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=32768,  # safe default for long context
        enforce_eager=True,   # recommended for multimodal models
    )

    processor = AutoProcessor.from_pretrained(args.model_path)
    log.info("LLM and processor loaded successfully")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )

    # ========================================
    # Load Questions & Resume Logic
    # ========================================
    questions_path = os.path.join(args.storage_path, "dataset", "Questions.json")
    with open(questions_path) as f:
        all_messages = json.load(f)

    log.info("Loaded %d questions", len(all_messages))

    # Resume from existing output
    if os.path.exists(args.output_file_path):
        with open(args.output_file_path) as f:
            outputs = json.load(f)
        # Keep only valid entries with "answer"
        outputs = [o for o in outputs if isinstance(o, dict) and "answer" in o]
    else:
        outputs = []

    already_done = len(outputs)
    messages = all_messages[already_done:]
    log.info("Resuming – %d already processed, %d remaining", already_done, len(messages))

    # ========================================
    # Batched Inference Loop
    # ========================================
    batch_requests = []

    for batch_start in tqdm(range(0, len(messages), args.batch_size), desc="vLLM Batches"):
        batch_messages = messages[batch_start:batch_start + args.batch_size]
        batch_requests.clear()

        for msg in batch_messages:
            try:
                conversation = [msg]

                # Extract metadata
                video_path = "UNKNOWN"
                question = ""
                for item in msg.get("content", []):
                    if item["type"] == "video":
                        video_path = item.get("video", "UNKNOWN")
                    if item["type"] == "text":
                        question = item.get("text", "")

                prompt = processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    conversation, return_video_kwargs=True
                )

                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs

                request = {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": video_kwargs or {},
                    "metadata": {
                        "video": video_path,
                        "question": question
                    }
                }
                batch_requests.append(request)

            except Exception as e:
                video_path = "UNKNOWN"
                try:
                    video_path = next((c["video"] for c in msg.get("content", []) if c["type"] == "video"), "UNKNOWN")
                except:
                    pass
                error_entry = {
                    "video": video_path,
                    "question": "",
                    "answer": f"[PREPROCESS ERROR] {traceback.format_exc()}"
                }
                outputs.append(error_entry if args.save_with_metadata else error_entry["answer"])
                log.error("Failed to prepare sample: %s", str(e))

        if not batch_requests:
            log.warning("No valid requests in this batch – skipping")
            continue

        # ========================================
        # vLLM Batched Generation
        # ========================================
        try:
            log.info("Submitting batch of %d requests to vLLM", len(batch_requests))
            vllm_outputs = llm.generate(batch_requests, sampling_params=sampling_params)

            for req, out in zip(batch_requests, vllm_outputs):
                generated_text = out.outputs[0].text.strip()
                metadata = req["metadata"]

                if args.save_with_metadata:
                    result = {
                        "video": metadata["video"],
                        "question": metadata["question"],
                        "answer": generated_text
                    }
                else:
                    result = generated_text

                outputs.append(result)

        except Exception as e:
            log.error("vLLM batch generation failed: %s", traceback.format_exc())
            for req in batch_requests:
                error_entry = {
                    "video": req["metadata"]["video"],
                    "question": req["metadata"]["question"],
                    "answer": f"[VLLM ERROR] {str(e)}"
                }
                outputs.append(error_entry if args.save_with_metadata else error_entry["answer"])

        # ========================================
        # Save intermediate results
        # ========================================
        with open(args.output_file_path, "w") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)

        log.info("Batch completed – total saved: %d", len(outputs))

    log.info("All done!")
    log.info("Final results with metadata saved to: %s", args.output_file_path)
    log.info("Total processed samples: %d", len(outputs))

    print("\nInference completed successfully!")
    print(f"Results saved to: {args.output_file_path}")
    print(f"Total entries: {len(outputs)}")