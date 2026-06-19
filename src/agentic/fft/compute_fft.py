"""
FFT Frequency Analysis for AI-Generated Video Detection

Analyzes the power spectrum of video frames to detect spectral fingerprints
left by AI generators. Real camera footage has characteristic sensor noise;
generated frames tend to be unnaturally clean or show periodic grid patterns.

Usage (standalone):
    python compute_fft.py --video_path <path> --output_dir <path>

Usage (as module):
    from fft.compute_fft import process_video_fft
    result = process_video_fft(video_path, output_dir)
"""

import argparse
import gc
import json
import numpy as np
from pathlib import Path

# Try torchvision first (consistent with RAFT tool), fall back to cv2
try:
    from torchvision.io import read_video, write_video
    import torch
    USE_TORCHVISION = True
except ImportError:
    import cv2
    USE_TORCHVISION = False


def load_frames(video_path: Path, max_frames: int = 60) -> list:
    """
    Load video frames as numpy arrays (H, W) grayscale.
    Samples uniformly if video has more frames than max_frames.
    """
    if USE_TORCHVISION:
        frames_tensor, _, metadata = read_video(str(video_path), output_format="TCHW")
        n = frames_tensor.shape[0]
        
        # Sample uniformly
        if n > max_frames:
            indices = np.linspace(0, n - 1, max_frames, dtype=int)
        else:
            indices = np.arange(n)
        
        gray_frames = []
        for i in indices:
            frame = frames_tensor[i].numpy()  # (C, H, W)
            # Convert to grayscale: 0.299R + 0.587G + 0.114B
            gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
            gray_frames.append(gray.astype(np.float64))
        
        return gray_frames
    else:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total > max_frames:
            indices = set(np.linspace(0, total - 1, max_frames, dtype=int))
        else:
            indices = set(range(total))
        
        gray_frames = []
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                gray_frames.append(gray)
        
        cap.release()
        return gray_frames


def compute_radial_profile(magnitude_spectrum: np.ndarray) -> np.ndarray:
    """
    Compute the azimuthally averaged radial power profile from a centered
    magnitude spectrum. Returns 1D array where index = frequency bin.
    """
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    
    # Create radius map
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    
    max_r = min(cy, cx)
    radial_profile = np.zeros(max_r)
    counts = np.zeros(max_r)
    
    # Mask to only include up to max_r
    mask = r < max_r
    r_masked = r[mask]
    vals_masked = magnitude_spectrum[mask]
    
    np.add.at(radial_profile, r_masked, vals_masked)
    np.add.at(counts, r_masked, 1)
    
    # Avoid division by zero
    counts[counts == 0] = 1
    radial_profile /= counts
    
    return radial_profile


def analyze_frame_fft(gray_frame: np.ndarray) -> dict:
    """
    Compute FFT analysis for a single grayscale frame.
    Returns dict with spectral metrics.
    """
    # Apply windowing to reduce edge artifacts
    h, w = gray_frame.shape
    window_y = np.hanning(h)
    window_x = np.hanning(w)
    window = np.outer(window_y, window_x)
    windowed = gray_frame * window
    
    # 2D FFT
    f_transform = np.fft.fft2(windowed)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # Log magnitude (avoid log(0))
    log_magnitude = np.log1p(magnitude)
    
    # Radial power profile
    radial = compute_radial_profile(log_magnitude)
    
    # Split into frequency bands
    n_bins = len(radial)
    low_end = int(n_bins * 0.1)
    mid_start = int(n_bins * 0.1)
    mid_end = int(n_bins * 0.5)
    high_start = int(n_bins * 0.5)
    
    low_freq_energy = np.mean(radial[1:low_end]) if low_end > 1 else 0  # skip DC
    mid_freq_energy = np.mean(radial[mid_start:mid_end]) if mid_end > mid_start else 0
    high_freq_energy = np.mean(radial[high_start:]) if high_start < n_bins else 0
    total_energy = np.mean(radial[1:])  # skip DC component
    
    # High-frequency ratio (key metric for generation detection)
    hf_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
    
    # Spectral slope (log-log fit of radial profile)
    # Natural images follow ~1/f power law; generated images deviate
    valid = radial[1:] > 0
    if np.sum(valid) > 10:
        freqs = np.arange(1, n_bins)
        log_freqs = np.log(freqs[valid])
        log_power = np.log(radial[1:][valid])
        # Linear fit in log-log space
        coeffs = np.polyfit(log_freqs, log_power, 1)
        spectral_slope = coeffs[0]
    else:
        spectral_slope = 0.0
    
    return {
        'low_freq_energy': float(low_freq_energy),
        'mid_freq_energy': float(mid_freq_energy),
        'high_freq_energy': float(high_freq_energy),
        'hf_ratio': float(hf_ratio),
        'spectral_slope': float(spectral_slope),
        'radial_profile': radial.tolist(),
    }


def compute_anisotropy(gray_frame: np.ndarray) -> dict:
    """
    Measure directional energy concentration in high-frequency FFT region.
    
    Cross-shaped artifacts from AI generators concentrate energy along specific
    angles (0°, 45°, 90°, 135°). Real footage distributes energy uniformly.
    
    Returns anisotropy score: higher = more directional (likely generated).
    Empirical ranges: real ~0.03-0.047, generated ~0.04-0.07
    """
    h, w = gray_frame.shape
    
    f = np.fft.fft2(gray_frame.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    
    cy, cx = h // 2, w // 2
    
    # Radial distance map
    Y, X = np.mgrid[:h, :w]
    R = np.sqrt((Y - cy)**2 + (X - cx)**2)
    max_r = min(cy, cx)
    
    # High-freq ring: 30%-90% of max radius
    inner_r = max_r * 0.3
    outer_r = max_r * 0.9
    ring_mask = (R >= inner_r) & (R <= outer_r)
    
    # Angle for each pixel relative to center
    angles = np.arctan2(Y - cy, X - cx)  # -pi to pi
    
    # Sample energy in angular bins
    n_bins = 72  # every 5 degrees
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    
    radial_energy = np.zeros(n_bins)
    for i in range(n_bins):
        angle_mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        combined = ring_mask & angle_mask
        if combined.any():
            radial_energy[i] = magnitude[combined].mean()
    
    if radial_energy.max() < 1e-10:
        return {'anisotropy': 0.0, 'peak_ratio': 0.0, 'axis_ratio': 0.0}
    
    # Anisotropy: coefficient of variation
    anisotropy = float(radial_energy.std() / (radial_energy.mean() + 1e-10))
    
    # Peak ratio: max / median
    median_e = np.median(radial_energy)
    peak_ratio = float(radial_energy.max() / (median_e + 1e-10))
    
    # Axis energy ratio: energy on cardinal+diagonal axes vs rest
    def angle_to_bin(deg):
        rad = deg * np.pi / 180
        return int((rad + np.pi) / (2 * np.pi) * n_bins) % n_bins
    
    axis_angles = [0, 45, 90, 135, -180, -135, -90, -45]
    axis_bins = [angle_to_bin(a) for a in axis_angles]
    non_axis_bins = [i for i in range(n_bins) if i not in axis_bins]
    
    axis_energy_mean = np.mean(radial_energy[axis_bins]) if axis_bins else 0
    non_axis_energy_mean = np.mean(radial_energy[non_axis_bins]) if non_axis_bins else 1
    axis_ratio = float(axis_energy_mean / (non_axis_energy_mean + 1e-10))
    
    return {
        'anisotropy': round(anisotropy, 4),
        'peak_ratio': round(peak_ratio, 4),
        'axis_ratio': round(axis_ratio, 4),
    }


def detect_periodic_artifacts(gray_frame: np.ndarray, threshold_std: float = 4.0) -> dict:
    """
    Detect periodic/grid artifacts in the spectrum that indicate
    generator architecture patterns (e.g., checkerboard from transposed convolutions).
    """
    h, w = gray_frame.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    windowed = gray_frame * window
    
    f_shift = np.fft.fftshift(np.fft.fft2(windowed))
    magnitude = np.abs(f_shift)
    log_mag = np.log1p(magnitude)
    
    # Compute radial profile and subtract it to find anomalous peaks
    radial = compute_radial_profile(log_mag)
    
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cy, cx)
    
    # Create expected spectrum from radial average
    expected = np.zeros_like(log_mag)
    for ri in range(max_r):
        mask = r == ri
        expected[mask] = radial[ri] if ri < len(radial) else 0
    
    # Residual: actual - expected
    residual = log_mag - expected
    
    # Only look at non-DC frequencies
    center_mask = r > 5  # skip very low frequencies
    residual_valid = residual[center_mask]
    
    mean_res = np.mean(residual_valid)
    std_res = np.std(residual_valid)
    
    # Count anomalous peaks
    if std_res > 0:
        anomalous = np.sum(residual_valid > mean_res + threshold_std * std_res)
        total = len(residual_valid)
        anomaly_ratio = anomalous / total
    else:
        anomaly_ratio = 0.0
        anomalous = 0
    
    return {
        'anomalous_peaks': int(anomalous),
        'anomaly_ratio': float(anomaly_ratio),
        'residual_std': float(std_res),
    }


def render_fft_frame(gray_frame: np.ndarray) -> np.ndarray:
    """
    Render a single frame's FFT magnitude spectrum as a color-mapped image.
    Returns an RGB uint8 array (H, W, 3) suitable for video encoding.
    """
    h, w = gray_frame.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    windowed = gray_frame * window

    f_shift = np.fft.fftshift(np.fft.fft2(windowed))
    magnitude = np.log1p(np.abs(f_shift))

    # Normalize to 0-255
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max > mag_min:
        normalized = ((magnitude - mag_min) / (mag_max - mag_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(magnitude, dtype=np.uint8)

    # Apply a colormap (hot: black → red → yellow → white)
    # Manual hot colormap to avoid matplotlib dependency
    r = np.clip(normalized * 3, 0, 255).astype(np.uint8)
    g = np.clip((normalized - 85) * 3, 0, 255).astype(np.uint8)
    b = np.clip((normalized - 170) * 3, 0, 255).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=-1)  # (H, W, 3)
    return rgb


def render_fft_video(video_path: Path, output_dir: Path, max_frames: int = 60) -> Path:
    """
    Render the FFT magnitude spectrum of each frame as a video.
    Returns the path to the output video file.
    """
    frames = load_frames(video_path, max_frames=max_frames)
    if len(frames) < 2:
        return None

    # Get original video fps
    fps = 15.0  # default
    if USE_TORCHVISION:
        try:
            _, _, metadata = read_video(str(video_path), output_format="TCHW", pts_unit="sec")
            fps = metadata.get('video_fps', 15.0)
        except Exception:
            pass
    else:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        cap.release()

    # Render each frame's FFT spectrum
    fft_frames = []
    for gray in frames:
        rgb = render_fft_frame(gray)
        fft_frames.append(rgb)

    output_path = output_dir / f"{video_path.stem}_fft.mp4"
    temp_path = output_dir / f"{video_path.stem}_fft_temp.mp4"

    # Write uncompressed first, then compress with ffmpeg (H.264, CRF 28)
    if USE_TORCHVISION:
        video_tensor = torch.from_numpy(np.stack(fft_frames))
        write_video(str(temp_path), video_tensor, fps=fps)
    else:
        h, w = fft_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (w, h))
        for frame_rgb in fft_frames:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()

    # Compress with ffmpeg (H.264, CRF 28 for small files)
    import subprocess
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', str(temp_path),
            '-c:v', 'libx264', '-crf', '28', '-preset', 'fast',
            '-pix_fmt', 'yuv420p', str(output_path)
        ], capture_output=True, check=True)
        temp_path.unlink()  # delete temp
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg not available, keep uncompressed
        temp_path.rename(output_path)

    del frames, fft_frames
    gc.collect()

    return output_path


def process_video_fft(video_path: Path, output_dir: Path, max_frames: int = 60) -> dict:
    """
    Full FFT analysis pipeline for a video.
    
    Returns a dict with:
    - summary statistics across all frames
    - per-frame metrics
    - text interpretation for the VLM agent
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load frames
    frames = load_frames(video_path, max_frames=max_frames)
    if len(frames) < 2:
        return {
            'error': f'Could not load enough frames from {video_path.name}',
            'interpretation': f'Error: Could not load frames from {video_path.name}'
        }
    
    # Analyze each frame
    frame_metrics = []
    artifact_detections = []
    anisotropy_scores = []
    
    for i, frame in enumerate(frames):
        metrics = analyze_frame_fft(frame)
        artifacts = detect_periodic_artifacts(frame)
        aniso = compute_anisotropy(frame)
        
        frame_metrics.append({
            'frame_idx': i,
            'hf_ratio': metrics['hf_ratio'],
            'spectral_slope': metrics['spectral_slope'],
            'high_freq_energy': metrics['high_freq_energy'],
            'anomaly_ratio': artifacts['anomaly_ratio'],
            'anomalous_peaks': artifacts['anomalous_peaks'],
            'anisotropy': aniso['anisotropy'],
        })
        
        artifact_detections.append(artifacts)
        anisotropy_scores.append(aniso)
    
    # Aggregate statistics
    hf_ratios = [m['hf_ratio'] for m in frame_metrics]
    slopes = [m['spectral_slope'] for m in frame_metrics]
    anomaly_ratios = [a['anomaly_ratio'] for a in artifact_detections]
    aniso_values = [a['anisotropy'] for a in anisotropy_scores]
    peak_ratios = [a['peak_ratio'] for a in anisotropy_scores]
    axis_ratios = [a['axis_ratio'] for a in anisotropy_scores]
    
    summary = {
        'num_frames_analyzed': len(frames),
        'anisotropy': {
            'mean': float(np.mean(aniso_values)),
            'std': float(np.std(aniso_values)),
            'max': float(np.max(aniso_values)),
        },
        'peak_ratio': {
            'mean': float(np.mean(peak_ratios)),
        },
        'axis_ratio': {
            'mean': float(np.mean(axis_ratios)),
        },
        'hf_ratio': {
            'mean': float(np.mean(hf_ratios)),
            'std': float(np.std(hf_ratios)),
            'min': float(np.min(hf_ratios)),
            'max': float(np.max(hf_ratios)),
        },
        'spectral_slope': {
            'mean': float(np.mean(slopes)),
            'std': float(np.std(slopes)),
        },
        'anomaly_ratio': {
            'mean': float(np.mean(anomaly_ratios)),
            'max': float(np.max(anomaly_ratios)),
        },
        'periodic_artifact_frames': sum(1 for a in anomaly_ratios if a > 0.001),
    }
    
    # Generate text interpretation for the agent
    interpretation = generate_interpretation(summary)
    
    # Render FFT spectrum video
    fft_video_path = render_fft_video(video_path, output_dir, max_frames=max_frames)

    # Save full results
    result = {
        'video': str(video_path.name),
        'fft_video': str(fft_video_path) if fft_video_path else None,
        'summary': summary,
        'frame_metrics': frame_metrics,
        'interpretation': interpretation,
    }
    
    output_path = output_dir / f"{video_path.stem}_fft_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Clean up
    del frames
    gc.collect()
    
    return result


def generate_interpretation(summary: dict) -> str:
    """
    Generate a human-readable interpretation of the FFT analysis
    that the VLM agent can use for reasoning.
    
    The PRIMARY metric is anisotropy (directional energy concentration).
    AI generators leave cross-shaped artifacts in the FFT spectrum that
    concentrate energy along specific angles. Real footage is uniform.
    
    Calibrated on Cosmos-Drive-Dreams dataset:
      Real footage:    anisotropy 0.034 - 0.047 (mean ~0.040)
      AI-generated:    anisotropy 0.041 - 0.068 (mean ~0.055)
      Threshold: 0.048 (tuned for balanced precision/recall)
    """
    lines = []
    n_frames = summary['num_frames_analyzed']
    
    # Primary metric: Anisotropy
    aniso_mean = summary['anisotropy']['mean']
    aniso_max = summary['anisotropy']['max']
    peak_ratio = summary['peak_ratio']['mean']
    axis_ratio = summary['axis_ratio']['mean']
    
    ANISO_THRESHOLD = 0.048
    
    lines.append(f"=== FFT Directional Energy Analysis ({n_frames} frames) ===")
    lines.append(f"")
    lines.append(f"PRIMARY METRIC — Anisotropy (directional energy concentration in high frequencies):")
    lines.append(f"  anisotropy = {aniso_mean:.4f}  (threshold: {ANISO_THRESHOLD})")
    lines.append(f"  peak_ratio = {peak_ratio:.4f}")
    lines.append(f"  axis_ratio = {axis_ratio:.4f}")
    lines.append(f"")
    lines.append(f"  Reference ranges (calibrated on driving video dataset):")
    lines.append(f"    Real footage:  0.034 - 0.047")
    lines.append(f"    AI-generated:  0.048 - 0.070")
    lines.append(f"")
    
    if aniso_mean >= ANISO_THRESHOLD:
        confidence = "HIGH" if aniso_mean >= 0.055 else "MODERATE"
        lines.append(f"  ➤ VERDICT HINT: GENERATED ({confidence} confidence)")
        lines.append(f"    Anisotropy {aniso_mean:.4f} > {ANISO_THRESHOLD} indicates directional")
        lines.append(f"    energy patterns (cross-shaped artifacts) typical of AI generators.")
    else:
        confidence = "HIGH" if aniso_mean <= 0.040 else "MODERATE"
        lines.append(f"  ➤ VERDICT HINT: REAL ({confidence} confidence)")
        lines.append(f"    Anisotropy {aniso_mean:.4f} ≤ {ANISO_THRESHOLD} indicates uniform energy")
        lines.append(f"    distribution consistent with natural camera footage.")
    
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFT Frequency Analysis for video")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=60)
    args = parser.parse_args()
    
    result = process_video_fft(
        Path(args.video_path),
        Path(args.output_dir),
        max_frames=args.max_frames
    )
    
    print(result['interpretation'])
