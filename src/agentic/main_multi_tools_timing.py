import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"   # or "5"

import json
import argparse
import time
from tqdm import tqdm
from typing import List, Dict
from pathlib import Path

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

from utils.raft import process_video_raft
from utils.sam import process_video_sam

################################
##### Global Configuration #####
################################
ROOT_DIR = Path("/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset")

# Build a filename → full Path cache once at import time (very fast, even for >50k videos)
print("Building Video cache")
_VIDEO_CACHE: dict[str, Path] = {}
for ext in ("*.mp4", "*.MP4"):
    for video_path in ROOT_DIR.rglob(ext):
        if video_path.is_file():
            _VIDEO_CACHE[video_path.name] = video_path
print("Video cache buit !")

def parse_args():
    parser = argparse.ArgumentParser(description="Agent VQA inference with Qwen3-VL-30B-A3B-Instruct + vLLM")
    parser.add_argument("--storage_path", type=str, default="/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval")
    parser.add_argument("--questions_file", type=str, default="dataset/Questions_raw.json",
                        help="JSON file containing the list of questions (same format as Aria benchmark)")
    parser.add_argument("--output_file_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_input_tokens", type=int, default=128000)

    return parser.parse_args()

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


def get_absolute_path(identifier):
    # ------------------------------------------------------------
    # 1. Resolve the identifier to an absolute Path
    # ------------------------------------------------------------
    if Path(identifier).exists() and Path(identifier).is_file():
        # Rare case: user gave an absolute path
        video_path = Path(identifier)
    elif '/' in identifier or '\\' in identifier:
        # Looks like a relative path → use directly
        video_path = ROOT_DIR / identifier
    else:
        # Only a filename → look it up in the cache
        if identifier not in _VIDEO_CACHE:
            raise FileNotFoundError(
                f"Video filename '{identifier}' not found in {ROOT_DIR}. "
                "Make sure the filename is exact (including extension) and unique."
            )
        video_path = _VIDEO_CACHE[identifier]

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    return video_path
        
###############################
##### Tool 1 : Optical Flow (RAFT)
###############################
@register_tool("get_motion_info")
class GetMotionInfo(BaseTool):
    description = """
    Generate the RGB video of an optical flow of a video to get motion information. 
    It also provides a mapping from colors to directions and real-world meanings. 
    """
    parameters: List[Dict] = [{
        "name": "video_path",
        "type": "string",
        "description": "Relative path of the video inside the dataset (e.g. 'cosmos-drive-dreams/crossing_red_lights/cross_red_002_Original.mp4')",
        "required": True
    }]

    def call(self, params: str, **kwargs) -> str:
        params = json.loads(params)
        identifier = params["video_path"].strip()
        video_path = get_absolute_path(identifier)

        # Build output directory: insert '/raft' right after 'final_dataset'
        parts = list(video_path.parts)
        idx = parts.index('final_dataset')
        parts.insert(idx + 1, 'raft')
        output_dir = Path(*parts).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        output_video_path = process_video_raft(video_path=video_path, output_dir=output_dir)

        return json.dumps({
            "motion_visualization_video": str(output_video_path),
        })


###############################
##### Tool 2 : SAM 3 Object Isolation
###############################
@register_tool("get_masks")
class GetMasks(BaseTool):
    description = """
    Isolates and extracts every individually tracked object (e.g., each car) from a driving video using SAM 3 video segmentation.
    For each detected object, it returns a clean, tightly cropped, aspect-ratio-preserved video clip (512×512, black background) 
    containing ONLY that object — with all background removed and only the frames where the object is visible.
    These clips are ideal for detecting AI-generation artifacts because they eliminate scene clutter and reveal 
    any unnatural deformation, melting, or wobbling of rigid objects like vehicles.
    Use this tool when you need to inspect individual cars frame-by-frame for physical consistency and rigidity.
    """

    parameters: List[Dict] = [
        {
            "name": "prompt",
            "type": "string",
            "description": "The object type to segment and track (e.g., 'car', 'truck', 'person').",
            "required": True,
        },
        {
            "name": "video_path",
            "type": "string",
            "description": "The path to the video file.",
            "required": True,
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        params = json.loads(params)
        identifier = params["video_path"].strip()
        video_path = get_absolute_path(identifier)

        # Build output directory: insert '/raft' right after 'final_dataset'
        parts = list(video_path.parts)
        idx = parts.index('final_dataset')
        parts.insert(idx + 1, 'sam')
        output_dir = Path(*parts).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        output_videos_path = process_video_sam(prompt=params.get('prompt'), 
                                              video_path=video_path, 
                                              output_dir=output_dir)


        # Instead of JSON, format as Markdown video tags so the agent "sees" them.
        # Qwen-Agent parser looks for ![video](path) to load the media.
        
        video_messages = []
        for i, p in enumerate(output_videos_path):
            # Convert path to string and ensure it's absolute if needed
            file_path = str(p)
            video_messages.append(f"Clip {i+1}: ![video]({file_path})")
            
        return "I have isolated the objects. Here are the video clips:\n" + "\n".join(video_messages)
        



###############################
##### Tool 3 : Output formatting
###############################
@register_tool("final_answer")
class FinalAnswer(BaseTool):
    description = """
    Use this tool **only once** at the very end of your reasoning.
    It formats the final evaluation of the video exactly as required.
    """
    parameters: List[Dict] = [
        {
            "name": "evaluation",
            "type": "string",
            "description": "Your detailed rationale/explanation in natural language (1–4 sentences).",
            "required": True
        },
        {
            "name": "answer",
            "type": "string",
            "description": "The exact final answer. "
                          "For Yes/No questions: 'Yes' or 'No'.\n"
                          "For Real/Generated: 'Real' or 'Generated'.\n"
                          "For MCQ: Depends on the possible answers.\n",
            "required": True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json.loads(params)

        evaluation = p["evaluation"].strip()
        answer = p["answer"].strip().capitalize()

       ## Save the output in a temp json file
        temp_path = os.path.join('/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/results', "temp_final_answer.json")
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump({'evaluation': evaluation, 'answer': answer}, f, indent=4, ensure_ascii=False)
        return "Your message is saved, thank you !"
    


if __name__ == "__main__":

    args = parse_args()

    if args.output_file_path is None:
        os.makedirs(os.path.join(args.storage_path, "results"), exist_ok=True)
        args.output_file_path = os.path.join(args.storage_path, "results", "Qwen3VL_Agent_results.json")

    # ========================================
    # Load questions + resume logic
    # ========================================
    questions_path = os.path.join(args.storage_path, args.questions_file)
    with open(questions_path) as f:
        questions = json.load(f)

    print(f"{len(questions)} questions loaded")

    if os.path.exists(args.output_file_path):
        with open(args.output_file_path) as f:
            outputs = json.load(f)
        # Keep only valid entries
        outputs = [o for o in outputs if isinstance(o, dict) and "answer" in o]
        questions = get_current_questions(questions, outputs)
    else:
        outputs = []

    processed = len(outputs)
    print(f"Resuming from {processed} already processed entries")

    ###############################
    ##### LLM & Agent Configuration
    ###############################
    llm_cfg = {
        'model_type': 'qwenvl_oai',
        'model': '/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/models/Qwen3-VL-30B-A3B-Instruct',
        'model_server': 'http://localhost:8000/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'top_p': args.top_p,
            'top_k': args.top_k,
            'temperature': args.temperature,
            'max_input_tokens': args.max_input_tokens,
        }
    }

    system_instruction = """
    You are an expert multimodal forensic and scene analysis agent. Your sole purpose is to answer the user's question accurately by following the mandatory Standard Operating Procedures (SOP) below.

    ### 1. Mandatory Tool Routing (SOP-A)

    You must first determine the user's intent to select the correct tool. **FAILURE TO ROUTE IS FORBIDDEN.** Do not answer based on the raw video alone if a tool is required.

    | User Intent / Keywords | Mandatory Tool Call | SOP to Follow |
    | :--- | :--- | :--- |
    | **Authenticity/Forensics:** "Real or generated?", "Artifacts?", "Is this fake?" | `get_masks` with prompt="car" | **SOP-B (AI Forensics)** |
    | **Motion/Violation:** "Traffic rule violation?", "Speed?", "Which direction?", "Movement?" | `get_motion_info` with video_path | **SOP-C (Motion/Violation)** |

    ---

    ### 2. AI Forensics Analysis (SOP-B: Required for `get_masks` output)

    If you call `get_masks` (AI Forensics), follow this high-priority, detailed protocol precisely.

    **SOP-B.1. Examine Isolated Clips (CRITICAL ANALYSIS):**
    * The tool returns **multiple** tightly cropped clips. **You must analyze every single clip** to form a judgment on the entire source video.
    * **MANDATORY DETAIL REQUIREMENT:** You **MUST** internally generate a detailed, point-form analysis for **EACH CLIP** before proceeding.
    **SOP-B.2. Artifact Detection Criteria (The "Forensic Scan"):**
    For each isolated car clip, check for physical impossibilities:
    - **Rigidity Violation:** Does the roof, door, or window frame wobble, breathe, or melt? 
    - **Texture Swimming:** Do the textures on the car surface "swim" while the car moves?
    - **Component Morphing:** Do wheels or mirrors change shape or merge into the body?


    **SOP-B.3. Final Verdict Logic:**
    * If you find **ANY** rigidity-violating artifact (Jelly Effect, Morphing, etc.) in **A SINGLE FRAME OF A SINGLE CLIP** to **ANSWER: Generated**.
    * Only if **ALL** isolated objects in **ALL** clips maintain geometric perfection and rigidity for the entire duration to **ANSWER: Real**.

    ---

    ### 3. Motion & Violation Analysis (SOP-C: Required for `get_motion_info` output)

    If you call `get_motion`, the output will be referred to as the **Optical Flow Video**. Follow these steps precisely:

    **SOP-C.1. Tool Call and Video Identification:**
    1.  Identify the original video provided in the user query (refer to it as **Video 1**).
    2.  Call `get_motion_info` using the **filename only** of Video 1.

    **SOP-C.2. Interpret Optical Flow Color Mapping:**
    The Optical Flow Video uses a specific color code:
    * **Pink/Magenta:** **Downward motion in image** (indicating **ego car moving forward**).
    * **Green/Yellow:** Upward motion in image (indicating ego car moving backward).
    * **Red/Blue:** Lateral motion. 
    * Brightness/Saturation indicates the magnitude (speed) of the motion.

    **SOP-C.3. Analysis and Priority Rule:**
    1.  Analyze the Optical Flow Video to detect **forward motion of the ego car** (focus on **Pink/Magenta patterns**).
    2.  Cross-reference this flow analysis with **Video 1** for context (e.g., traffic light color, stop line).
    3.  **PRIORITY RULE:** If interpretations differ (e.g., Video 1 looks static, but flow shows Pink/Magenta), **RELY PRIMARILY ON THE OPTICAL FLOW VIDEO** for motion confirmation relative to the stop line and traffic light.

    **SOP-C.4. Final Answer Formulation:**
    * Provide a clear, reasoned answer, citing evidence from both videos.
    * The final response must explicitly describe what the Optical Flow Video showed, and **prioritize optical flow for motion confirmation**.

    ---

    ### 4. Final Output Formatting

    * You **MUST** end by calling the tool `final_answer`.
    * Your `evaluation` field **MUST** synthesize the detailed evidence from your internal clip-by-clip inspection (SOP-B) or the motion analysis (SOP-C). State exactly which clip/object failed the rigidity test, or how the optical flow confirmed the motion.
    """

    bot = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        function_list=['get_motion_info', 'get_masks', 'final_answer'],
        name='Driving Scene Analyst'
    )

    ##############################
    ############  Run ############
    ##############################    
    temp_path = os.path.join('/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/results', "temp_final_answer.json")
    for item in tqdm(questions, desc='Agent working'):
        content = item["content"]
        video_item = next(c for c in content if c["type"] == "video")
        text_item = next(c for c in content if c["type"] == "text")

        video_path = video_item["video"]
        question = text_item["text"]
        
        initial_message: List[Dict] = [
        {
            'role': 'user',
            'content': [
                {
                    "video": video_path
                },
                {
                    "text": question
                }
            ]
        }
        ]
        
        #### VLM processing
        responses = []
        # Start timing the inference
        inference_start_time = time.time()
        for rsp in bot.run(messages=initial_message):
            responses.extend(rsp)
        # End timing the inference
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time


        #### Output formating
        ### Open the temp output to get the evaluation and the answer and feed it to the main output json
        if os.path.exists(temp_path):
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            evaluation = data.get('evaluation', '')
            answer = data.get('answer', '')
            # Optionally, remove the temp file after reading
            os.remove(temp_path)
        else:
            evaluation = ''
            answer = ''
        #### output formating
        content = item["content"]
        video_item = next(c for c in content if c["type"] == "video")
        text_item = next(c for c in content if c["type"] == "text")
        video_path = video_item["video"]
        question = text_item["text"]
        outputs.append({
            "video": video_path,
            "question": question,
            "history" : rsp,
            "evaluation" : evaluation,
            "answer": answer,
            "inference_time": inference_time,
        })
        with open(args.output_file_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)