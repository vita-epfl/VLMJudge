import os
import gc
import cv2
import torch
import imageio
import numpy as np
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from pathlib import Path

MODEL_PATH = '/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/models/sam3'
FINAL_SIZE = 512          # Final output will be 512×512
PADDING_RATIO = 0.25             # 25% extra padding around bbox
SMOOTHING = 0.6                  # Higher = smoother crop (0.0–1.0)

# ----------------------------
# Helper: letterbox to square with black borders
# ----------------------------
def letterbox_to_square(img_np, target_size=FINAL_SIZE):
    h, w = img_np.shape[:2]
    if h == w:
        return cv2.resize(img_np, (target_size, target_size), interpolation=cv2.INTER_AREA) if 'cv2' in globals() else np.array(Image.fromarray(img_np).resize((target_size, target_size), Image.LANCZOS))
    
    # Compute scaling factor to fit inside target_size × target_size
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize while preserving aspect ratio
    resized = np.array(Image.fromarray(img_np).resize((new_w, new_h), Image.LANCZOS))
    
    # Create black canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas


def process_video_sam(prompt : str, video_path: Path, output_dir: Path):
    
    print("Loading SAM3")
    model = Sam3VideoModel.from_pretrained(MODEL_PATH).to('cuda', dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained(MODEL_PATH)
    print("Sam loaded")

    # Create folder for saving masked videos
    output_dir = str(output_dir / video_path.stem)
    os.makedirs(output_dir, exist_ok=True)

    # Load video frames and timestamps from local file
    video_frames, metadata = load_video(str(video_path), return_timestamps=True)

    fps = metadata["fps"]
    H, W = video_frames[0].shape[:2]

    # Initialize video inference session
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device='cuda',
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )

    # Add text prompt to detect and track objects
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=prompt,
    )

    # Process all frames in the video
    outputs_per_frame = {}
    all_ids = set()

    for out in model.propagate_in_video_iterator(inference_session, max_frame_num_to_track=130):
        processed = processor.postprocess_outputs(inference_session, out)
        idx = out.frame_idx
        outputs_per_frame[idx] = processed
        all_ids.update(processed["object_ids"].tolist())

    frames_sorted = sorted(outputs_per_frame.keys())


    # ----------------------------
    # Per-object: only keep frames where object is visible
    # ----------------------------
    for obj_id in sorted(all_ids):
        visible_frames = []        # Only frames with actual content
        prev_crop = None

        for idx in frames_sorted:
            outputs = outputs_per_frame[idx]
            ids_this_frame = outputs["object_ids"].tolist()

            if obj_id not in ids_this_frame:
                prev_crop = None
                continue  # Skip black frames entirely

            local_idx = ids_this_frame.index(obj_id)
            box = outputs["boxes"][local_idx].cpu().numpy()
            mask = (outputs["masks"][local_idx].float().cpu().numpy() > 0.0).astype(np.uint8)

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            w_box = x2 - x1
            h_box = y2 - y1
            pad_x = int(w_box * PADDING_RATIO)
            pad_y = int(h_box * PADDING_RATIO)

            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(W, x2 + pad_x)
            cy2 = min(H, y2 + pad_y)

            # Smooth crop window
            if prev_crop is not None:
                px1, py1, px2, py2 = prev_crop
                cx1 = SMOOTHING * px1 + (1 - SMOOTHING) * cx1
                cy1 = SMOOTHING * py1 + (1 - SMOOTHING) * cy1
                cx2 = SMOOTHING * px2 + (1 - SMOOTHING) * cx2
                cy2 = SMOOTHING * py2 + (1 - SMOOTHING) * cy2
            cx1, cy1, cx2, cy2 = map(int, [cx1, cy1, cx2, cy2])
            prev_crop = (cx1, cy1, cx2, cy2)

            # Crop + mask
            frame = video_frames[idx]
            region = frame[cy1:cy2, cx1:cx2]
            mask_region = mask[cy1:cy2, cx1:cx2]
            masked_region = region * mask_region[..., np.newaxis]

            # Letterbox to square
            final_frame = letterbox_to_square(masked_region, FINAL_SIZE)
            visible_frames.append(final_frame)

        # Only save if object appeared at least once
        if len(visible_frames) == 0:
            continue

        # Save clean, trimmed video
        out_path = os.path.join(output_dir, f"object_{obj_id}_clean.mp4")
        writer = imageio.get_writer(
            out_path,
            format='ffmpeg',
            fps=fps,
            codec='libx264',
            output_params=['-crf', '18', '-pix_fmt', 'yuv420p']
        )
        for f in visible_frames:
            writer.append_data(f)
        writer.close()

    # cleaning the memory
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect() # Force garbage collection

    output_videos_path = [f"{output_dir}/{video}" for video in os.listdir(output_dir)]
    return output_videos_path