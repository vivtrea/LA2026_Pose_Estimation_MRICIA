"""
pipeline.py — Main entry point for UAV pose estimation via MR-ICIA.

Usage:
    python pipeline.py --data_dir /path/to/vpair/query \
                       --fx 868.99 --fy 868.99 \
                       --cx 525.0  --cy 399.0

VPAIR camera intrinsics (from dataset metadata):
    fx = 868.99, fy = 868.99, cx = 525.0, cy = 399.0
    Image resolution: 1024 x 800

If you don't have intrinsics yet, use --estimate_K to approximate from FoV.
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from homography import build_homography, compose_warp, propagate_params
from icia import compute_jacobian_and_hessian, icia_step
from pose import recover_pose
from pyramid import build_pyramid


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_frames(data_dir: str) -> list:
    """
    Load all grayscale frames from a directory, sorted by filename.
    Supports .png, .jpg, .jpeg.

    Args:
        data_dir: Path to directory containing aerial image frames.

    Returns:
        List of grayscale float32 numpy arrays, one per frame.
    """
    data_dir = Path(data_dir)
    extensions = {".png", ".jpg", ".jpeg"}
    paths = sorted([p for p in data_dir.iterdir()
                    if p.suffix.lower() in extensions])

    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    frames = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not read image: {p}")
        frames.append(img.astype(np.float32))

    print(f"Loaded {len(frames)} frames from {data_dir}")
    return frames


def load_ground_truth(pose_file: str) -> list:
    """
    Load VPAIR ground-truth 6-DoF poses from a CSV file.
    Expected columns: frame_id, tx, ty, tz, roll, pitch, yaw

    Args:
        pose_file: Path to CSV file with ground-truth poses.

    Returns:
        List of dicts with keys: tx, ty, tz, roll, pitch, yaw.
        Returns empty list if file does not exist.
    """
    if not os.path.exists(pose_file):
        print(f"Warning: pose file not found at {pose_file}. "
              "RMSE evaluation will be skipped.")
        return []

    poses = []
    with open(pose_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append({k: float(v) for k, v in row.items()
                          if k != "frame_id"})
    return poses


# ---------------------------------------------------------------------------
# Camera matrix
# ---------------------------------------------------------------------------

def build_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Construct the 3x3 camera intrinsic matrix K.

    Args:
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point (image centre).

    Returns:
        3x3 float64 camera matrix.
    """
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)


def estimate_K_from_fov(image_w: int, image_h: int,
                         fov_deg: float = 90.0) -> np.ndarray:
    """
    Approximate K from horizontal field of view when intrinsics are unknown.
    Useful for getting started; replace with calibrated values when available.

    Args:
        image_w:  Image width in pixels.
        image_h:  Image height in pixels.
        fov_deg:  Horizontal field of view in degrees (default 90°).

    Returns:
        3x3 approximate camera matrix.
    """
    fov_rad = np.radians(fov_deg)
    fx = (image_w / 2.0) / np.tan(fov_rad / 2.0)
    return build_K(fx, fx, image_w / 2.0, image_h / 2.0)


# ---------------------------------------------------------------------------
# Template cropping
# ---------------------------------------------------------------------------

def crop_template(img: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    """
    Crop the central region of an image for use as the ICIA template.
    The paper uses ~80% of the image (Section II-B).

    Args:
        img:   Full grayscale image.
        ratio: Fraction of image to keep (0.8 = 80%).

    Returns:
        Cropped central region.
    """
    h, w = img.shape
    dh = int(h * (1 - ratio) / 2)
    dw = int(w * (1 - ratio) / 2)
    return img[dh:h - dh, dw:w - dw]


# ---------------------------------------------------------------------------
# MR-ICIA core loop
# ---------------------------------------------------------------------------

def mr_icia(
    template:  np.ndarray,
    current:   np.ndarray,
    levels:    int   = 4,
    max_iters: int   = 100,
    tol:       float = 1e-5,
) -> np.ndarray:
    """
    Multi-Resolution Inverse Compositional Image Alignment (MR-ICIA).
    Implements Algorithm 1 from the project report / Martinez et al. (2011).

    Finds the homography Hp that best aligns `current` to `template` by:
      1. Building image pyramids for both images.
      2. At each level (coarse to fine):
           a. Precompute Jacobian J and Hessian H from template gradient.
           b. Iteratively solve for delta_p and update p via composition.
           c. Propagate parameters to the next finer level.
      3. Return the final projective homography Hp.

    Args:
        template:  Grayscale template image (float32).
        current:   Grayscale current image to align (float32).
        levels:    Number of pyramid levels (paper uses 4).
        max_iters: Maximum ICIA iterations per level (paper uses 100).
        tol:       Convergence threshold for ||delta_p|| (paper uses 1e-5).

    Returns:
        Hp: 3x3 projective homography matrix.
    """
    T_pyr = build_pyramid(template, levels)  # pyramid.py
    I_pyr = build_pyramid(current,  levels)  # pyramid.py

    p = np.zeros(8, dtype=np.float64)  # start at identity warp

    for level in range(levels):  # index 0 = coarsest
        T_l = T_pyr[level]
        I_l = I_pyr[level]

        # Precompute J and H once per pyramid level (ICIA advantage)
        J_flat, H_mat = compute_jacobian_and_hessian(T_l)  # icia.py
        H_inv = np.linalg.inv(H_mat)

        prev_mse = np.inf
        no_improve_count = 0

        for iteration in range(max_iters):
            delta_p, mse = icia_step(T_l, I_l, p, J_flat, H_inv)  # icia.py
            p = compose_warp(p, delta_p)                           # homography.py

            # Termination criterion T1: parameter increment below threshold
            if np.linalg.norm(delta_p) < tol:
                break

            # Termination criterion T2: no improvement in MSE for 10 iters
            if mse >= prev_mse:
                no_improve_count += 1
                if no_improve_count >= 10:
                    break
            else:
                no_improve_count = 0
            prev_mse = mse

        # Propagate parameters to next finer level
        if level < levels - 1:
            p = propagate_params(p)  # homography.py

    return build_homography(p)  # homography.py


# ---------------------------------------------------------------------------
# RMSE evaluation
# ---------------------------------------------------------------------------

def compute_rmse(estimated: list, ground_truth: list) -> dict:
    """
    Compute RMSE between estimated and ground-truth Euler angles.

    Args:
        estimated:    List of (roll, pitch, yaw) tuples in degrees.
        ground_truth: List of dicts with keys roll, pitch, yaw.

    Returns:
        Dict with RMSE values for roll, pitch, yaw.
    """
    n = min(len(estimated), len(ground_truth))
    if n == 0:
        return {}

    est = np.array(estimated[:n])
    gt  = np.array([[g["roll"], g["pitch"], g["yaw"]]
                    for g in ground_truth[:n]])

    rmse = np.sqrt(np.mean((est - gt) ** 2, axis=0))
    return {"roll": rmse[0], "pitch": rmse[1], "yaw": rmse[2]}

# In pipeline.py — replace load_ground_truth() with this:

def load_vpair_poses(pose_file: str) -> list:
    """
    Load VPAIR poses from poses_query.txt.
    Columns: filepath, x, y, z, undulation, roll, pitch, yaw
    x, y, z are ECEF coordinates in metres.
    roll, pitch, yaw are in degrees.
    """
    poses = []
    with open(pose_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append({
                "filepath": row["filepath"],
                "x":     float(row["x"]),
                "y":     float(row["y"]),
                "z":     float(row["z"]),
                "roll":  np.degrees(float(row["roll"])),
                "pitch": np.degrees(float(row["pitch"])),
                "yaw":   np.degrees(float(row["yaw"])),
            })
    return poses


def compute_rmse_vpair(estimated_angles: list, gt_relative: list) -> dict:
    """
    Compare estimated Euler angles from MR-ICIA against relative
    ground truth rotations computed by compute_relative_angles().
    """
    n = min(len(estimated_angles), len(gt_relative))

    angle_errors = []

    for i in range(n):
        est_roll,  est_pitch,  est_yaw  = estimated_angles[i]
        gt_roll,   gt_pitch,   gt_yaw   = gt_relative[i]  # unpack tuple

        angle_errors.append([
            est_roll  - gt_roll,
            est_pitch - gt_pitch,
            est_yaw   - gt_yaw,
        ])

    errors = np.array(angle_errors)
    rmse   = np.sqrt(np.mean(errors**2, axis=0))

    return {
        "roll_rmse":  rmse[0],
        "pitch_rmse": rmse[1],
        "yaw_rmse":   rmse[2],
    }

def compute_relative_angles(gt_poses: list) -> list:
    """
    Convert absolute GT poses into relative frame-to-frame rotations.
    Assumes angles are already in degrees (converted in load_vpair_poses).
    """
    relative = []
    for i in range(len(gt_poses) - 1):
        d_roll  = gt_poses[i+1]["roll"]  - gt_poses[i]["roll"]
        d_pitch = gt_poses[i+1]["pitch"] - gt_poses[i]["pitch"]
        d_yaw   = gt_poses[i+1]["yaw"]   - gt_poses[i]["yaw"]
        relative.append((d_roll, d_pitch, d_yaw))
    return relative


def plot_errors_over_time(estimated_angles: list, gt_relative: list, output_path: str = "errors.png"):
    """
    Plot per-frame angular error for roll, pitch, yaw over time.
    """
    n = min(len(estimated_angles), len(gt_relative))

    frames  = list(range(n))
    e_roll  = [abs(estimated_angles[i][0] - gt_relative[i][0]) for i in range(n)]
    e_pitch = [abs(estimated_angles[i][1] - gt_relative[i][1]) for i in range(n)]
    e_yaw   = [abs(estimated_angles[i][2] - gt_relative[i][2]) for i in range(n)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Per-frame angular error vs ground truth", fontsize=13)

    for ax, errors, label, color in zip(
        axes,
        [e_roll, e_pitch, e_yaw],
        ["Roll error (°)", "Pitch error (°)", "Yaw error (°)"],
        ["steelblue", "darkorange", "seagreen"]
    ):
        ax.plot(frames, errors, color=color, linewidth=0.8)
        ax.axhline(y=5.0, color="red", linestyle="--", linewidth=0.8, label="5° threshold")
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Error plot saved to {output_path}")
    plt.show()

def has_sufficient_texture(img: np.ndarray, threshold: float = 30.0) -> bool:
    """
    Check if image has enough texture for ICIA to work.
    Uses gradient magnitude as a proxy for texture richness.
    threshold=5.0 works well for 8-bit aerial images.
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    # print(float(np.mean(gradient_magnitude)))
    return float(np.mean(gradient_magnitude)) > threshold
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UAV pose estimation via MR-ICIA image alignment."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing sequential aerial image frames."
    )
    parser.add_argument(
        "--pose_file", type=str, default="",
        help="CSV file with ground-truth poses (optional, for RMSE eval)."
    )
    parser.add_argument(
        "--fx", type=float, default=0.0,
        help="Focal length x in pixels (VPAIR: 868.99)."
    )
    parser.add_argument(
        "--fy", type=float, default=0.0,
        help="Focal length y in pixels (VPAIR: 868.99)."
    )
    parser.add_argument(
        "--cx", type=float, default=0.0,
        help="Principal point x (VPAIR: 525.0)."
    )
    parser.add_argument(
        "--cy", type=float, default=0.0,
        help="Principal point y (VPAIR: 399.0)."
    )
    parser.add_argument(
        "--estimate_K", action="store_true",
        help="Estimate K from FoV instead of using --fx/fy/cx/cy."
    )
    parser.add_argument(
        "--fov", type=float, default=90.0,
        help="Horizontal FoV in degrees, used only with --estimate_K."
    )
    parser.add_argument(
        "--levels", type=int, default=4,
        help="Number of pyramid levels (default: 4)."
    )
    parser.add_argument(
        "--max_iters", type=int, default=100,
        help="Max ICIA iterations per pyramid level (default: 100)."
    )
    parser.add_argument(
        "--output", type=str, default="results.csv",
        help="Output CSV file for estimated poses."
    )
    args = parser.parse_args()

    # --- Load frames ---
    frames = load_frames(args.data_dir)

    # --- Build camera matrix K ---
    if args.estimate_K:
        h, w = frames[0].shape
        K = estimate_K_from_fov(w, h, args.fov)
        print(f"Using estimated K from FoV={args.fov}°:\n{K}")
    elif args.fx > 0:
        K = build_K(args.fx, args.fy, args.cx, args.cy)
        print(f"Using provided K:\n{K}")
    else:
        # VPAIR default intrinsics
        K = build_K(750.626, 750.263, 402.410, 292.988)
        print(f"Using VPAIR default K:\n{K}")

    # --- Load ground-truth poses (optional) ---
    gt_poses = load_vpair_poses(args.pose_file) if args.pose_file else []

    # --- Run MR-ICIA on consecutive frame pairs ---
    estimated_angles = []
    results = []

    for i in range(len(frames) - 1):
        T_full = frames[i]
        I_full = frames[i + 1]

        # Crop central 80% of template (as in paper)
        T = crop_template(T_full, ratio=0.8)
        I = crop_template(I_full, ratio=0.8)

        if not has_sufficient_texture(T) or not has_sufficient_texture(I):
            print(f"Frame {i:04d}: low texture, skipping")
            if estimated_angles:
                roll, pitch, yaw = estimated_angles[-1]
            else:
                roll, pitch, yaw = 0.0, 0.0, 0.0
            estimated_angles.append((roll, pitch, yaw))
            continue
        # Run MR-ICIA to get projective homography
        Hp = mr_icia(T, I, levels=args.levels, max_iters=args.max_iters)

        # Recover pose from homography
        R, t, (roll, pitch, yaw) = recover_pose(Hp, K)

        estimated_angles.append((roll, pitch, yaw))
        results.append({
            "frame":  i,
            "roll":   roll,
            "pitch":  pitch,
            "yaw":    yaw,
            "tx":     float(t[0]),
            "ty":     float(t[1]),
            "tz":     float(t[2]),
        })

        print(f"Frame {i:04d} -> {i+1:04d} | "
              f"roll={roll:7.3f}°  pitch={pitch:7.3f}°  yaw={yaw:7.3f}°")

    # --- RMSE evaluation ---
    if gt_poses:
        gt_relative = compute_relative_angles(gt_poses)
        rmse = compute_rmse_vpair(estimated_angles, gt_relative)
        print("\n--- RMSE vs ground truth ---")
        print(f"  Roll:  {rmse['roll_rmse']:.4f}°")
        print(f"  Pitch: {rmse['pitch_rmse']:.4f}°")
        print(f"  Yaw:   {rmse['yaw_rmse']:.4f}°")
        plot_errors_over_time(estimated_angles, gt_relative, output_path="moving_rmse.png")

    # --- Save results to CSV ---
    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
