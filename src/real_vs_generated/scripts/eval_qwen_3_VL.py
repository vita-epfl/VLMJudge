#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import traceback
from typing import List, Any

from tqdm import tqdm
import logging

# vLLM imports (≥0.11.0)
from vllm import LLM, SamplingParams

# Qwen-VL specific imports
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# ========================================
# Argument Parser
# ========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Batched VQA inference with Qwen3-VL-30B-A3B-Instruct + vLLM")
    parser.add_argument("--model_path", type=str,
                        help="Path or HF repo of the Qwen-VL model, e.g. Qwen/Qwen3-VL-30B-A3B-Instruct",
                        default="/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/models/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of GPUs (tensor parallel)")
    parser.add_argument("--storage_path", type=str, default="/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval")
    parser.add_argument("--questions_file", type=str, default="dataset/Questions.json",
                        help="JSON file containing the list of questions (same format as Aria benchmark)")
    parser.add_argument("--output_file_path", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=15,
                        help="Effective batch size (will be limited by GPU memory; 24–40 typical on 4x80GB)")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=1.5)

    return parser.parse_args()

# ========================================
# Helper function
# ========================================

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    #print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

def get_current_questions(messages,output):
    unique_couples = []
    # Creation of the unique (video,question) couples
    for response in output:
        current_video = response['video']
        current_question = response['question']
        if (current_video,current_question) not in unique_couples:
            unique_couples.append((current_video,current_question))
    # Only keep the ones which are not answered
    messages_to_answer = []

    for message in messages:
        current_video = message['content'][0]['video']
        current_question = message['content'][1]['text']
        if (current_video,current_question) not in unique_couples:
            messages_to_answer.append(message)
    return messages_to_answer



if __name__ == '__main__':

    args = parse_args()

    if args.output_file_path is None:
        os.makedirs(os.path.join(args.storage_path, "results"), exist_ok=True)
        args.output_file_path = os.path.join(args.storage_path, "results", "Qwen3VL_results.json")


    # ========================================
    # Logging
    # ========================================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.storage_path, "qwen3vl_vllm.log"))
        ],
    )
    log = logging.getLogger(__name__)


    # ========================================
    # 1. Load Model + Processor
    # ========================================
    log.info(f"Loading {args.model_path} with tensor_parallel_size={args.tensor_parallel_size}")

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        seed=0,
        mm_encoder_tp_mode="data",
        #enable_expert_parallel=True,
        trust_remote_code=True,
        enforce_eager=True,                  # safer for very large models
        max_model_len=32768,                 # Qwen3-VL supports long contexts
        gpu_memory_utilization=0.94,
        max_num_batched_tokens=65536,
        # multimodal config is auto-detected for Qwen-VL
    )

    processor = AutoProcessor.from_pretrained(
        args.model_path,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        stop_token_ids=[]
    )

    

    log.info("Model and processor loaded successfully")

    # ========================================
    # 2. Load questions + resume logic
    # ========================================
    # SURGICAL PATCH: allow absolute --questions_file (for realgen ablation)
    questions_path = args.questions_file if os.path.isabs(args.questions_file) \
        else os.path.join(args.storage_path, args.questions_file)
    with open(questions_path) as f:
        questions = json.load(f)

    log.info(f"{len(questions)} questions loaded")

    if os.path.exists(args.output_file_path):
        with open(args.output_file_path) as f:
            outputs = json.load(f)
        # Keep only valid entries
        outputs = [o for o in outputs if isinstance(o, dict) and "answer" in o]
        questions = get_current_questions(questions, outputs)
    else:
        outputs = []

    processed = len(outputs)
    log.info(f"Resuming from {processed} already processed entries")


    # ========================================
    # 3. Main batched inference loop
    # ========================================
    batch_outputs: List[Any] = []
    total_questions = len(questions) + processed

    pbar = tqdm(
        total=total_questions,
        initial=processed,
        desc=f"Progress (0/{total_questions})",
        unit="question",
        colour="green",
        dynamic_ncols=True,
    )

    for batch_start in range(0, len(questions), args.batch_size):
        batch = questions[batch_start:batch_start + args.batch_size]

        messages = []
        metadata_list = []

        for item in batch:
            try:
                content = item["content"]
                video_item = next(c for c in content if c["type"] == "video")
                text_item = next(c for c in content if c["type"] == "text")

                video_path = video_item["video"]
                question = text_item["text"]

                # Build the official chat format expected by Qwen-VL
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_path, "nframes" : args.num_frames},
                            {"type": "text", "text": question},
                        ],
                    }
                ]

                messages.append(message)
                metadata_list.append((video_path, question))

            except Exception as e:
                err = {
                    "video": video_path if 'video_path' in locals() else "UNKNOWN",
                    "question": question if 'question' in locals() else "",
                    "error": traceback.format_exc(),
                }
                pbar.update(1)
                continue


        inputs = [prepare_inputs_for_vllm(message, processor) for message in messages]


        if not inputs:
            log.warning("Entire batch failed → skipping")
            continue

        # ========================================
        # vLLM generation (multimodal)
        # ========================================
        try:
            vllm_outputs = llm.generate(inputs, sampling_params=sampling_params)


            for vllm_out, (video_path, q) in zip(vllm_outputs, metadata_list):
                answer = vllm_out.outputs[0].text.strip()
                batch_outputs.append({
                    "video": video_path,
                    "question": q,
                    "answer": answer,
                })


        except Exception as e:
            log.error(f"vLLM generation failed for batch: {traceback.format_exc()}")

        # Update progress & save
        successful = len([x for x in batch_outputs[-len(metadata_list):] if "answer" in x])
        pbar.update(successful)
        pbar.set_description(f"Progress ({processed + len(outputs) + len(batch_outputs)}/{total_questions})")

        outputs.extend(batch_outputs)
        with open(args.output_file_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)

        log.info(f"Batch saved – {len(batch_outputs)} new entries (total: {len(outputs)})")
        batch_outputs.clear()

    pbar.set_description("Inference completed!")
    pbar.close()

    log.info(f"All done! Results saved to: {args.output_file_path}")
    print(f"\nFinished! Total processed questions: {len(outputs) + processed}")