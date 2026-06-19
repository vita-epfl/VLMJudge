import os
import json
import re
import polars as pl

def extract_question_from_prompt(prompt_text):
    """
    Extract the actual question from a prompt template.
    
    The prompt format is typically:
    "You will be given a video and a question couple.
    ...
    Question: <actual question> ?
    
    Provide your feedback."
    
    Or for MCQ questions:
    "...
    Question: <actual question>
    Possible answers : 1 2 3 
    
    Provide your feedback."
    
    Or with "The answer is one of the following":
    "...
    Question: <actual question>
    The answer is one of the following : 1 2 3"
    """
    # Try to extract question after "Question:" and before various terminators
    # The terminators can be:
    # - "Provide your feedback"
    # - "Possible answers"
    # - "The answer is one of the following"
    
    # First, find the "Question:" part
    question_match = re.search(r'Question:\s*', prompt_text)
    if not question_match:
        return prompt_text.strip()
    
    # Start extracting from after "Question:"
    start_pos = question_match.end()
    remaining_text = prompt_text[start_pos:]
    
    # Find the end of the question (before any terminator)
    terminators = [
        r'\n\s*Provide your feedback',
        r'\n\s*Possible answers',
        r'\n\s*The answer is one of the following',
    ]
    
    earliest_end = len(remaining_text)
    for terminator in terminators:
        match = re.search(terminator, remaining_text, re.IGNORECASE)
        if match:
            earliest_end = min(earliest_end, match.start())
    
    question = remaining_text[:earliest_end].strip()
    return question


def clean_results_timing(raw_results):
    """
    Clean results from timing JSON files.
    
    Expected keys in each entry:
    - video: path to video file
    - question: the question asked
    - answer: the VLM's answer
    - inference_time: time taken for inference
    """
    path_to_remove = '/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset/'
    cleaned_results = []
    
    for el in raw_results:
        # Extract the actual question from the prompt template
        question = extract_question_from_prompt(el["question"])
        
        # Clean the answer
        answer = el["answer"]
        
        # Clean the video path
        video_path = re.sub(path_to_remove, '', el["video"])
        
        # Get inference time
        inference_time = el.get("inference_time", None)
        
        new_answer = {
            'video': video_path,
            'question': question,
            'raw_output': answer,
            'vlm_answer': answer,
            'inference_time': inference_time
        }
        cleaned_results.append(new_answer)
    
    return cleaned_results


def enrich_with_ground_truth_single(
    cleaned_items,
    questions_dd,
    questions_p1
):
    """
    Enriches a list of cleaned VLM outputs with the ground-truth answer and model name.
    
    Parameters
    ----------
    cleaned_items : list[dict]
        List of dictionaries containing at least 'video', 'question', 'raw_output', 'vlm_answer', 'inference_time'.
    questions_dd : list[dict]
        Source list for videos whose path starts with 'cosmos-drive-dreams/'.
    questions_p1 : list[dict]
        Source list for videos whose path starts with 'cosmos-predict1/'.

    Returns
    -------
    list[dict]
        The original items enriched with 'model' and 'ground_truth' keys.
    """
    # Build fast lookup dictionaries: (video_basename, question) → ground_truth answer
    lookup_dd = {}
    lookup_p1 = {}

    for entry in questions_dd:
        video_key = entry['video']                     # e.g. 'negative/agg_tk_030_Foggy.mp4'
        for qa in entry.get('qa_pairs', []):
            key = (video_key, qa['question'])
            lookup_dd[key] = {
                'answer': qa['answer'],
                'category': qa.get('category')  # safely handle missing category
            }

    for entry in questions_p1:
        video_key = entry['video']
        for qa in entry.get('qa_pairs', []):
            key = (video_key, qa['question'])
            lookup_p1[key] = {
                'answer': qa['answer'],
                'category': qa.get('category')  # safely handle missing category
            }

    enriched = []
    for item in cleaned_items:
        full_video_path = item['video']                # e.g. 'cosmos-drive-dreams/negative/agg_tk_030_Foggy.mp4'
        question = item['question']

        # Determine model and base video name
        if full_video_path.startswith('cosmos-drive-dreams/'):
            model = 'cosmos-drive-dreams'
            base_video = full_video_path[len('cosmos-drive-dreams/'):]  # remove prefix
            lookup = lookup_dd
        elif full_video_path.startswith('cosmos-predict1/'):
            model = 'cosmos-predict1'
            base_video = full_video_path[len('cosmos-predict1/'):]
            lookup = lookup_p1
        else:
            # Fallback if prefix is unexpected
            print(f"Unknown video path prefix in: {full_video_path}")
            continue

        key = (base_video, question)
        info = lookup.get(key)

        if info is None:
            print(f"Entry not found for video='{base_video}', question='{question}' in model '{model}'")
        else:
            # Create enriched item
            enriched_item = item.copy()
            enriched_item['model'] = model
            enriched_item['video'] = base_video  # optional: keep only the relative path (as in ground-truth lists)
            enriched_item['ground_truth'] = info['answer']
            enriched_item['category'] = info['category']

            enriched.append(enriched_item)

    return enriched


def enriched_list_to_polars_df_single(enriched_list):
    """
    Convert the enriched list of dictionaries into a Polars DataFrame.
    
    Expected columns after enrichment:
    - model          : str   (e.g., 'cosmos-drive-dreams')
    - video          : str   (relative path, e.g., 'negative/agg_tk_030_Foggy.mp4')
    - question       : str
    - raw_output     : str
    - vlm_answer     : str   (e.g., 'Real', 'Generated')
    - ground_truth   : str   (e.g., 'Real', 'Generated')
    - inference_time : float (time taken for inference)
    
    The function also adds a convenient boolean column 'correct' for quick accuracy metrics.
    """
    if not enriched_list:
        # Return an empty DataFrame with the expected schema if the list is empty
        return pl.DataFrame({
            "model": pl.Utf8,
            "video": pl.Utf8,
            "question": pl.Utf8,
            "raw_output": pl.Utf8,
            "vlm_answer": pl.Utf8,
            "ground_truth": pl.Utf8,
            "category": pl.Utf8,
            "inference_time": pl.Float64,
            "correct": pl.Boolean,
        })

    df = pl.DataFrame(enriched_list)

    # Ensure consistent categorical types for answers (improves performance & memory)
    df = df.with_columns([
        pl.col("vlm_answer").cast(pl.Utf8).replace({"Real": "Real", "Generated": "Generated"}),  # normalise if needed
        pl.col("ground_truth").cast(pl.Utf8),
        pl.col("model").cast(pl.Categorical),
        pl.col("video").cast(pl.Utf8),
        pl.col("question").cast(pl.Utf8),
        pl.col("raw_output").cast(pl.Utf8),
        pl.col("category").cast(pl.Categorical),
        pl.col("inference_time").cast(pl.Float64)
    ])

    # Add a boolean column indicating whether the VLM answer matches the ground truth
    df = df.with_columns(
        (pl.col("vlm_answer") == pl.col("ground_truth")).alias("correct")
    )

    return df


def save_df(df, output_path):
    """Save a Polars DataFrame to a parquet file."""
    df.write_parquet(output_path)


if __name__ == "__main__":
    ### Loading
    root_dir = "./dataset/dataset_final"
    results_dir = f"{root_dir}/results_timing"
    questions_dd_path = f"{root_dir}/questions_cosmos-drive-dreams.json"
    questions_p1_path = f"{root_dir}/questions_cosmos-predict1.json"
    output_dir = f"{root_dir}/results_timing_analyzed"

    # Load ground truth questions
    with open(questions_dd_path, 'r') as f:
        questions_dd = json.load(f)

    with open(questions_p1_path, 'r') as f:
        questions_p1 = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each JSON file in results_timing
    for result_file in os.listdir(results_dir):
        if not result_file.endswith('.json'):
            continue
        
        model_name = re.sub(r'_results\.json$', '', result_file)
        print(f"\nProcessing {model_name}...")
        
        result_path = f"{results_dir}/{result_file}"
        with open(result_path, 'r') as f:
            raw_results = json.load(f)
        
        # Clean the results
        cleaned_results = clean_results_timing(raw_results)
        
        # Getting the ground truth
        enriched_results = enrich_with_ground_truth_single(cleaned_results, questions_dd, questions_p1)
        
        # Transforming to a polars dataframe
        df = enriched_list_to_polars_df_single(enriched_results)
        
        # Overall accuracy
        if len(df) > 0:
            accuracy = df['correct'].mean()
            avg_time = df['inference_time'].mean()
            print(f"Overall accuracy for {model_name}: {accuracy:.2%}" if accuracy is not None else f"Overall accuracy for {model_name}: N/A")
            print(f"Average inference time for {model_name}: {avg_time:.4f}s" if avg_time is not None else f"Average inference time for {model_name}: N/A")
        
        # Save to parquet
        output_path = f"{output_dir}/{model_name}_analyzed.parquet"
        save_df(df, output_path)
        print(f"Saved to {output_path}")

    print("\nDone!")
