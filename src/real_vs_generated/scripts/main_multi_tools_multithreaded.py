import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"   # or "5"

import json
import multiprocessing
import argparse
import sys
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
from filelock import FileLock

# SURGICAL PATCH: this file lives in realgen_ablation/scripts/ instead of
# agentic/, so make the `utils` package (sibling of the original) importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agentic"))

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


def get_current_questions(messages, output_raw):
    # Clean output
    path_to_add = '/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset/'
    output = []
    for el in output_raw:
        el['video'] = path_to_add + el["model"] + "/" + el['video']
        output.append(el)
    # Build set of answered (video, question) for fast lookup
    answered = set()
    for response in output:
        video = response['video']
        question = response['question']  # Already cleaned via regex
        answered.add((video, question))
    
    # Or if multiple per video, but since unique, set per video
    # But for matching, better per video dict of sets
    from collections import defaultdict
    answered_questions = defaultdict(set)
    for response in output:
        video = response['video']
        question = response['question']
        answered_questions[video].add(question)
    
    messages_to_answer = []
    for message in messages:
        current_video = message['content'][0]['video']
        current_question = message['content'][1]['text']  # Original
        
        if current_video not in answered_questions:
            messages_to_answer.append(message)
            continue
        
        # Check if any answered question for this video is substring of current
        matched = False
        for ans_q in answered_questions[current_video]:
            if ans_q in current_question:  # cleaned in original
                matched = True
                break
        
        if not matched:
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

    def __init__(self, cfg=None): # <-- Accept 'cfg' argument
        super().__init__(cfg)      # <-- Pass 'cfg' to the parent's __init__
        # Each process gets its own temp file
        self.temp_dir = tempfile.mkdtemp(prefix="final_answer_")
        self.temp_path = os.path.join(self.temp_dir, "result.json")

    def call(self, params: str, **kwargs) -> str:
        p = json.loads(params)

        evaluation = p["evaluation"].strip()
        answer = p["answer"].strip().capitalize()

        # Save to process-local temp file
        with open(self.temp_path, 'w', encoding='utf-8') as f:
            json.dump({'evaluation': evaluation, 'answer': answer}, f, ensure_ascii=False)

        return "Your message is saved, thank you !"
    
# ===============================
# Worker function (runs in each process)
# ===============================
def process_single_item(item: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    # Rebuild agent inside process (required for vLLM + CUDA isolation)
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
        system_message=system_instruction,  # Make sure this is defined globally or passed
        function_list=['get_motion_info', 'get_masks', 'final_answer'],
        name='Driving Scene Analyst'
    )

    content = item["content"]
    video_item = next(c for c in content if c["type"] == "video")
    text_item = next(c for c in content if c["type"] == "text")
    video_path = video_item["video"]
    question = text_item["text"]

    messages = [{
        'role': 'user',
        'content': [{"video": video_path}, {"text": question}]
    }]

    responses = []
    try:
        for rsp in bot.run(messages=messages):
            responses.extend(rsp)
    except Exception as e:
        return {
            "video": video_path,
            "question": question,
            "history": responses,
            "evaluation": f"ERROR during inference: {str(e)}",
            "answer": "ERROR",
            "error": True
        }

    # Read result from process-local temp file
    tool_instance = bot.function_map['final_answer']
    temp_path = tool_instance.temp_path

    evaluation = ""
    answer = ""
    if os.path.exists(temp_path):
        try:
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                evaluation = data.get('evaluation', '')
                answer = data.get('answer', '')
        except:
            evaluation = "Failed to parse final_answer"
            answer = "ERROR"
    else:
        evaluation = "final_answer tool was not called"
        answer = "MISSING"

    return {
        "video": video_path,
        "question": question,
        "history": rsp,
        "evaluation": evaluation,
        "answer": answer
    }
# ===============================
# Main: Multiprocessing Launcher
# ===============================
def main():
    # ----------------------------------------------------
    # CRITICAL FIX: Set multiprocessing start method to 'spawn'
    # This must be the first thing related to multiprocessing.
    # ----------------------------------------------------
    try:
        if multiprocessing.get_start_method() != 'spawn':
            # This check prevents setting it if it's already set or the OS defaults to spawn (like Windows)
            multiprocessing.set_start_method('spawn', force=True) 
            print("Set multiprocessing start method to 'spawn' for CUDA compatibility.")
    except RuntimeError:
        # Happens if set_start_method is called after a pool has been created.
        # Since you call it at the start of main(), this should be rare.
        pass

    parser = argparse.ArgumentParser(description="Parallel Agent VQA Inference")
    parser.add_argument("--storage_path", type=str, default="/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval")
    parser.add_argument("--questions_file", type=str, default="dataset/Questions_raw.json")
    parser.add_argument("--output_file_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_input_tokens", type=int, default=128000)
    parser.add_argument("--num_workers", type=int, default=10, help="Number of parallel processes (adjust to your GPU setup)")
    args = parser.parse_args()

    if args.output_file_path is None:
        os.makedirs(os.path.join(args.storage_path, "results"), exist_ok=True)
        args.output_file_path = os.path.join(args.storage_path, "results", "Qwen3VL_Agent_results.json")

    lock_file = args.output_file_path + ".lock"
    output_lock = FileLock(lock_file)

    # Load questions and resume
    # SURGICAL PATCH: allow absolute --questions_file (for realgen ablation)
    questions_path = args.questions_file if os.path.isabs(args.questions_file) \
        else os.path.join(args.storage_path, args.questions_file)
    with open(questions_path) as f:
        all_questions = json.load(f)

    if os.path.exists(args.output_file_path):
        with output_lock:
            with open(args.output_file_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        #existing = [o for o in existing if isinstance(o, dict) and "answer" in o and o.get("answer") not in ["ERROR", "MISSING"]]
        #done_pairs = {(o["video"], o["question"]) for o in existing}
        #questions = [q for q in all_questions if (q["content"][0]["video"], q["content"][1]["text"]) not in done_pairs]
        questions = get_current_questions(all_questions, existing)
        print(f"Resuming: {len(existing)} completed, {len(questions)} remaining")
    else:
        questions = all_questions
        print(f"Starting fresh: {len(questions)} questions")

    if len(questions) == 0:
        print("All questions already processed!")
        return

    # Process in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_single_item, item, args)
            for item in questions
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Overall Progress"):
            result = future.result()

            # Atomically append result
            with output_lock:
                if os.path.exists(args.output_file_path):
                    with open(args.output_file_path, 'r', encoding='utf-8') as f:
                        current_data = json.load(f)
                else:
                    current_data = []

                current_data.append(result)

                with open(args.output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(current_data, f, indent=4, ensure_ascii=False)

    print(f"All done! Results saved to {args.output_file_path}")


if __name__ == "__main__":
    # Define system_instruction globally or import it
    # It must be available in worker processes (define it here or in a module)
    

    main()