import json

storage_path = "/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset"

def build_hint_mapping(standard_questions_path):
    """
    Parses the standard_questions.json (with hints) to create a lookup dictionary.
    Key: (question_text, asked_context)
    Value: List of secondary questions
    """
    try:
        with open(standard_questions_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {standard_questions_path}")
        return {}

    mapping = {}
    
    # The structure is a list of dictionaries: [{"Category": [questions...]}, ...]
    for category_dict in data:
        for category_name, questions_list in category_dict.items():
            for q in questions_list:
                # We use both the text and the 'asked' field as a unique key
                # to distinguish between similar questions in different contexts
                q_text = q.get('question', '').strip()
                
                # Get the hints if they exist
                hints = q.get('secondary_questions', [])
                
                if hints:
                    mapping[q_text] = hints
                    
    return mapping

def create_llm_messages(questions, ai_model, hint_mapping):
    """
    Processes the DataFrame to create LLM messages with injected Hint Questions.
    """
    
    # --- TEMPLATES ---
    base_prompt_template = """You will be given a video and a main question. To ensure accuracy, you are also provided with a set of "Hint Questions" that you must answer first.

    Your task is to analyze the video and follow this two-step reasoning process:
    1. Answer the Hint Questions to gather objective evidence from the video.
    2. Use those answers to determine the final answer to the Main Question.

    Hint Questions:
    {secondary_questions}

    Provide your feedback as follows:

    Feedback:::
    Evaluation: (First, write the answer to each Hint Question. Then, explain your reasoning for the Main Question based on those facts.)
    Answer: (your final answer, as a Yes or No)

    You MUST provide values for 'Evaluation:' and 'Answer:' in your answer.

    Now here is the Main Question.

    Question: {question}

    Provide your feedback."""

    generated_prompt_template = """You will be given a video and a main question. To ensure accuracy, you are also provided with a set of "Hint Questions" that you must answer first.

    Your task is to analyze the video and follow this two-step reasoning process:
    1. Answer the Hint Questions to gather objective evidence from the video.
    2. Use those answers to determine the final answer to the Main Question which should either be Real or Generated.

    Hint Questions:
    {secondary_questions}

    Provide your feedback as follows:

    Feedback:::
    Evaluation: (First, write the answer to each Hint Question. Then, explain your reasoning for the Main Question based on those facts.)
    Answer: (your answer, as a Real or a Generated)

    You MUST provide values for 'Evaluation:' and 'Answer:' in your answer.

    Now here is the Main Question.

    Question: {question}

    Provide your feedback."""

    MCQ_prompt_template = """You will be given a video and a main question. To ensure accuracy, you are also provided with a set of "Hint Questions" that you must answer first.

    Your task is to analyze the video and follow this two-step reasoning process:
    1. Answer the Hint Questions to gather objective evidence from the video.
    2. Use those answers to determine the final answer to the Main Question which should be one of the possible answers of the question.

    Hint Questions:
    {secondary_questions}

    Provide your feedback as follows:

    Feedback:::
    Evaluation: (First, write the answer to each Hint Question. Then, explain your reasoning for the Main Question based on those facts.)
    Answer: (your answer, picked from one of the possible answers)

    You MUST provide values for 'Evaluation:' and 'Answer:' in your answer.

    Now here is the question and the possible answers.

    Question: {question}
    Possible answers : {possible_answers}

    Provide your feedback."""

    messages = []
    
    # Iterate over each row of the dataframe
    for row in questions:
        if 'qa_pairs' not in row.keys():
            continue
        video_path = row['video']
        qa_pairs = row['qa_pairs']

        if 'nOP1blfMCTg_48624_14b' in video_path:
            continue
        
        # Iterate over each dictionary in the qa_pairs list
        for qa_pair in qa_pairs:
            category = qa_pair["category"]
            if 'Visual understanding' in category:
                continue
            question_text = qa_pair['question']
            q_type = qa_pair["type"]
            # We try to get 'asked' from the pair to match the key in our mapping
            # If the dataset doesn't have 'asked', this might return None/Empty

            # --- RETRIEVE HINTS ---
            # Try to find specific hints for this question + context
            hints = hint_mapping.get(question_text.strip(), [])
            
            # Formatting the hints list into a string
            if hints:
                formatted_hints = "\n".join([f"- {h}" for h in hints])
            else:
                # Fallback if no hints are found in the mapping
                formatted_hints = "- Describe the scene in detail.\n- Look for any anomalies."

            # --- SELECT TEMPLATE ---
            if "generated" in question_text.lower() and q_type != 'MCQ':
                template = generated_prompt_template
                prompt_content = template.format(
                    question=question_text, 
                    secondary_questions=formatted_hints
                )
            elif q_type == 'MCQ':
                template = MCQ_prompt_template
                possible_answers = " ".join(qa_pair["possible_answers"])
                prompt_content = template.format(
                    question=question_text, 
                    possible_answers=possible_answers,
                    secondary_questions=formatted_hints
                )
            else:
                template = base_prompt_template
                prompt_content = template.format(
                    question=question_text,
                    secondary_questions=formatted_hints
                )

            # Create the message dictionary
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{storage_path}/{ai_model}/{video_path}"
                    },
                    {"type": "text", "text": prompt_content},
                ]
            }
            messages.append(message)
    
    return messages


# Usage
if __name__ == "__main__":
    input_path_dd = "./dataset/dataset_final/questions_cosmos-drive-dreams.json"
    input_path_p2 = "./dataset/dataset_final/questions_cosmos-predict1.json"
    
    # PATH TO THE FILE WITH HINT QUESTIONS (The one created in previous step)
    standard_questions_path = "./data_preparation/resources/hint_questions.json" 

    # 1. Build the Hint Map
    hint_map = build_hint_mapping(standard_questions_path)
    print(f"Loaded {len(hint_map)} hint configurations.")

    # 2. Load Datasets
    with open(input_path_dd, 'r') as f:
        questions_dd = json.load(f)
    with open(input_path_p2, 'r') as f:
        questions_p2 = json.load(f)
    
    if questions_dd is not None and questions_p2 is not None:
        # 3. Create messages with hints
        llm_messages_dd = create_llm_messages(questions_dd, 'cosmos-drive-dreams', hint_map)
        llm_messages_p2 = create_llm_messages(questions_p2, 'cosmos-predict1', hint_map)
        llm_messages = llm_messages_dd + llm_messages_p2
      
        output_file = './dataset/dataset_final/Questions_with_hint.json'
        
        try:
            with open(output_file, 'w') as f:
                json.dump(llm_messages, f, indent=4)
            print(f"Successfully saved {len(llm_messages)} messages to {output_file}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")