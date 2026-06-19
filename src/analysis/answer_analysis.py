import os
import re
import json
import polars as pl
from typing import List, Dict, Any


def clean_results(raw_results):
    path_to_remove = '/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset/'
    cleaned_results = {}
    for model in raw_results.keys():
        current_result = results[model]
        cleaned_results[model] = []
        for el in current_result:
            # Clean the question
            text_question = el["question"]
            match = re.search(r"Question:\s*(.*?)\s*\n", text_question)
            if match:
                question = match.group(1).strip()
            else:
                question = 'NO MATCH !'
            # Clean the answer
            text_answer = el["answer"]
            match = re.search(r"Answer:\s*(.*?)(?:\s*<\|im_end\|>|$|\n)", text_answer, re.DOTALL)
            if match:
                answer = match.group(1).strip().capitalize()
            else:
                answer = 'NO MATCH !'
            # Clean the video path
            video_path = re.sub(path_to_remove,'',el["video"])
            new_answer = {'video' : video_path, 
                          'question' : question, 
                          'raw_output' : el['answer'], 
                          'vlm_answer' : answer}
            cleaned_results[model].append(new_answer)
    return cleaned_results


def enrich_with_ground_truth_single(
    cleaned_items: List[Dict[str, Any]],
    questions_dd: List[Dict[str, Any]],
    questions_p1: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enriches a list of cleaned VLM outputs with the ground-truth answer and model name.
    
    Parameters
    ----------
    cleaned_items : list[dict]
        List of dictionaries containing at least 'video', 'question', 'raw_output', 'vlm_answer'.
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
            raise ValueError(f"Unknown video path prefix in: {full_video_path}")

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

def enrich_with_ground_truth(cleaned_results, questions_dd, questions_p1):
    enriched_results = {}
    for model in cleaned_results.keys():
        enriched_results[model] = enrich_with_ground_truth_single(cleaned_results[model], questions_dd, questions_p1)
    return enriched_results


def enriched_list_to_polars_df_single(enriched_list: List[Dict[str, Any]]) -> pl.DataFrame:
    """
    Convert the enriched list of dictionaries into a Polars DataFrame.
    
    Expected columns after enrichment:
    - model          : str   (e.g., 'cosmos-drive-dreams')
    - video          : str   (relative path, e.g., 'negative/agg_tk_030_Foggy.mp4')
    - question       : str
    - raw_output     : str
    - vlm_answer     : str   (e.g., 'Real', 'Generated')
    - ground_truth   : str   (e.g., 'Real', 'Generated')
    
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
            "category": pl.Utf8,      # ← new column
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
        pl.col("category").cast(pl.Categorical)
    ])

    # Add a boolean column indicating whether the VLM answer matches the ground truth
    df = df.with_columns(
        (pl.col("vlm_answer") == pl.col("ground_truth")).alias("correct")
    )

    return df

def enriched_dict_to_polars_df(enriched_results):
    enriched_df = {}
    for model in enriched_results.keys():
        new_key = re.sub('results','analyzed',model)
        enriched_df[new_key] = enriched_list_to_polars_df_single(enriched_results[model])
    return enriched_df

def save_df_dict(df_dict,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for model in df_dict.keys():
        df_dict[model].write_parquet(f'{output_dir}/{model}.parquet')


if __name__=="__main__":
    ### Loading
    root_dir = "./dataset/dataset_final"
    results_dir = f"{root_dir}/results"
    questions_dd_path = f"{root_dir}/questions_cosmos-drive-dreams.json"
    questions_p1_path = f"{root_dir}/questions_cosmos-predict1.json"

    with open(questions_dd_path, 'r') as f:
        questions_dd = json.load(f)

    with open(questions_p1_path, 'r') as f:
        questions_p1 = json.load(f)


    results = {}
    for result in os.listdir(results_dir):
        name = re.sub(".json",'',result)
        path = f"{results_dir}/{result}"
        with open(path, 'r') as f:
            results[name] = json.load(f)

    # Clean the results
    results = clean_results(results)

    # Getting the ground truth
    results_enriched = enrich_with_ground_truth(results, questions_dd, questions_p1)

    # Transforming to a polars dataframe
    df_dict = enriched_dict_to_polars_df(results_enriched)
    
    # Overall accuracy
    for model in df_dict.keys():
        print(f"\nOverall accuracy for {model}: {df_dict[model]['correct'].mean():.2%}")

    # Saving
    output_dir = f"{root_dir}/results_analyzed"
    save_df_dict(df_dict, output_dir)
    


