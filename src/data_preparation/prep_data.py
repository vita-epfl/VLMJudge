import json

storage_path="/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset"


def create_llm_messages(questions, ai_model):
    """
    Processes the Polars DataFrame to create a list of LLM message dictionaries.

    Args:
        dataframe (pl.DataFrame): The input DataFrame with 'id' and 'qa_pairs' columns.
    
    Returns:
        list: A list of message dictionaries ready to be saved as JSON.
    """
    base_prompt_template = """You will be given a video and a question couple.
    The question is here to evaluate your performances.
    Your task is to analyze the video and then answer by yes or no to the question.

    Provide your feedback as follows:

    Feedback:::
    Evaluation: (your rationale for the question, as a text)
    Answer: (your answer, as a yes or a no)

    You MUST provide values for 'Evaluation:' and 'Answer:' in your answer.

    Now here is the question.

    Question: {question}

    Provide your feedback."""

    generated_prompt_template = """You will be given a video and a question couple.
    The question is here to evaluate your performances.
    Your task is to analyze the video and then answer by Real or Generated to the question.

    Provide your feedback as follows:

    Feedback:::
    Evaluation: (your rationale for the question, as a text)
    Answer: (your answer, as a Real or a Generated)

    You MUST provide values for 'Evaluation:' and 'Answer:' in your answer.

    Now here is the question.

    Question: {question}

    Provide your feedback."""

    MCQ_prompt_template = """You will be given a video and a question couple.
    The question is here to evaluate your performances.
    Your task is to analyze the video and then choose one of the possible answers of the question.

    Provide your feedback as follows:

    Feedback:::
    Evaluation: (your rationale for the question, as a text)
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
            question = qa_pair['question']
            type = qa_pair["type"]
            
            # Format the prompt with the extracted question
            if "generated" in question and type != 'MCQ':
                prompt = generated_prompt_template.format(question=question)
            elif type == 'MCQ':
                possible_answers = ""
                for a in qa_pair["possible_answers"]:
                    possible_answers += f"{a} "
                prompt = MCQ_prompt_template.format(question=question, possible_answers=possible_answers)
            else:
                prompt = base_prompt_template.format(question=question)

            # Create the message dictionary
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{storage_path}/{ai_model}/{video_path}"
                    },
                    {"type": "text", "text": prompt},
                ]
            }
            messages.append(message)
    
    return messages


# Usage
if __name__ == "__main__":
    input_path_dd = "./dataset/dataset_final/questions_cosmos-drive-dreams.json"
    input_path_p2 = "./dataset/dataset_final/questions_cosmos-predict1.json"
    # opening the json
    with open(input_path_dd, 'r') as f:
        questions_dd = json.load(f)
    with open(input_path_p2, 'r') as f:
        questions_p2 = json.load(f)
    
    if questions_dd is not None and questions_p2 is not None:
        # Create the messages
        llm_messages_dd = create_llm_messages(questions_dd, 'cosmos-drive-dreams')
        llm_messages_p2 = create_llm_messages(questions_p2, 'cosmos-predict1')
        llm_messages = llm_messages_dd + llm_messages_p2
      
        # Define the output file path
        output_file = './dataset/dataset_final/Questions.json'
        
        # Save the messages to a JSON file
        try:
            with open(output_file, 'w') as f:
                json.dump(llm_messages, f, indent=4)
            print(f"Successfully saved {len(llm_messages)} messages to {output_file}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")