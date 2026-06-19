import gc
import torch
from pathlib import Path
# Using write_video since it's compatible with torchvision 0.23.0
from torchvision.io import read_video, write_video 
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.utils import flow_to_image

# --- Configuration and Setup ---

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model Weights and Transforms
WEIGHTS = Raft_Large_Weights.DEFAULT
TRANSFORMS = WEIGHTS.transforms()

def preprocess(img1, img2):
    """
    Preprocesses two images (frames) for the RAFT model.
    """
    # Note: Resize dimensions must be divisible by 8
    target_size = [520, 960] 
    
    img1 = F.resize(img1, size=target_size, antialias=False)
    img2 = F.resize(img2, size=target_size, antialias=False)
    
    return TRANSFORMS(img1, img2)


def process_video_raft(video_path: Path, output_dir: Path):
    """
    Processes a single video file, calculates optical flow, and 
    writes the flow images to a new MP4 video using torchvision.io.write_video.
    """

    # Load the model
    print("Loading RAFT model...")
    model = raft_large(weights=WEIGHTS, progress=False).to(DEVICE)
    model.eval()
    print("Model loaded.")
    
    # 1. Read the video frames
    try:
        frames, _, metadata = read_video(str(video_path), output_format="TCHW")
        fps = metadata['video_fps']
        
    except Exception as e:
        # Returning None indicates failure, allowing the overall TQDM bar to continue
        print(f"Error reading video {video_path.name}: {e}")
        return None

    num_frames = frames.shape[0]
    if num_frames < 2:
        print(f"Skipping {video_path.name}: Not enough frames for flow calculation.")
        return None

    # List to store the output flow images
    all_flow_frames = []
    
    # 2. Flow Calculation Loop (NO TQDM HERE)
    with torch.no_grad():
        for i in range(num_frames - 1):
            img1, img2 = frames[i], frames[i + 1]
            
            # Preprocess and prepare for model
            img1_pre, img2_pre = preprocess(img1, img2)
            img1_pre = img1_pre.unsqueeze(0).to(DEVICE)
            img2_pre = img2_pre.unsqueeze(0).to(DEVICE)

            # Predict flow
            list_of_flows = model(img1_pre, img2_pre)
            predicted_flow = list_of_flows[-1][0]
            
            # Convert flow (2, H, W) to RGB image (3, H, W) in [0, 255]
            flow_img_tensor = flow_to_image(predicted_flow).cpu()

            # write_video requires [H, W, C] layout, so we permute (C, H, W) -> (H, W, C)
            flow_img_hwc = flow_img_tensor.permute(1, 2, 0)
            
            all_flow_frames.append(flow_img_hwc)

    # 3. Final Video Write
    output_filename = output_dir / f"flow_{video_path.stem}.mp4"
    
    # Stack all frames: List of (H, W, C) -> Tensor[T, H, W, C]
    video_array = torch.stack(all_flow_frames)

    # write_video takes: filename, video_array (T, H, W, C), and fps
    write_video(
        filename=str(output_filename), 
        video_array=video_array, 
        fps=fps,
        video_codec="libx264" # Standard MP4 codec
    )

    # --- GPU Resource Cleanup ---
    del model # Remove the reference to the large model object
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Release cached memory back to the GPU
    gc.collect() # Optional: Force Python garbage collection
    # --- End Cleanup ---
    
    return output_filename