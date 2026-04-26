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
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from homography import build_homography, compose_warp, propagate_params
from icia import compute_jacobian_and_hessian, icia_step
from pose import recover_pose, rotation_to_euler, euler_to_rotation
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

def load_vpair_poses(pose_file: str) -> list:
    """
    Load absolute VPAIR poses from `poses_query.txt`.

    Expected columns:
        filepath, x, y, z, undulation, roll, pitch, yaw

    Positions are stored in ECEF metres. Angles are stored in radians in the
    dataset and converted to degrees here.

    Args:
        pose_file: Path to the VPAIR pose file.

    Returns:
        List of pose dictionaries with absolute position and Euler angles.
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


def compute_rmse_vpair(
    estimated_step_angles: list,
    ground_truth_angles: list,
) -> dict:
    """
    Compute Euler-angle RMSE against aligned absolute ground truth.

    The first estimate in the current pipeline corresponds to frame 1 because
    it is recovered from motion between frames 0 and 1.
    """
    n = min(len(estimated_step_angles), len(ground_truth_angles))

    angle_errors = []

    for i in range(n):
        est_roll,  est_pitch,  est_yaw  = estimated_step_angles[i]
        gt_roll,   gt_pitch,   gt_yaw   = ground_truth_angles[i]

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

def plot_errors_over_time(
    estimated_step_angles: list,
    ground_truth_angles: list,
    output_path: str = "errors.png",
):
    """
    Plot per-frame absolute angular error for roll, pitch, and yaw.

    Args:
        estimated_step_angles: Sequence of estimated Euler-angle tuples in
            degrees.
        ground_truth_angles: Sequence of GT Euler-angle tuples in degrees,
            aligned to the same frame indices as `estimated_step_angles`.
        output_path: Destination path for the PNG figure.
    """
    n = min(len(estimated_step_angles), len(ground_truth_angles))

    frames  = list(range(n))
    e_roll  = [abs(estimated_step_angles[i][0] - ground_truth_angles[i][0]) for i in range(n)]
    e_pitch = [abs(estimated_step_angles[i][1] - ground_truth_angles[i][1]) for i in range(n)]
    e_yaw   = [abs(estimated_step_angles[i][2] - ground_truth_angles[i][2]) for i in range(n)]

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
    if args.pose_file:
        gt_poses = load_vpair_poses(args.pose_file)
        gt_rotations = []
        gt_positions = []
        gt_absolute_angles = []

        for p in gt_poses:
            R = euler_to_rotation(p["roll"], p["pitch"], p["yaw"])
            t = np.array([p["x"], p["y"], p["z"]])

            gt_rotations.append(R)
            gt_positions.append(t)
            gt_absolute_angles.append((p["roll"], p["pitch"], p["yaw"]))

        R_global = gt_rotations[0]
        t_global = gt_positions[0]

    else:
        gt_poses = []
        R_global = np.eye(3)
        t_global = np.zeros(3)
    # --- Run MR-ICIA on consecutive frame pairs ---
    estimated_step_angles = []
    step_pose_results = []
    absolute_pose_results = []

    for i in range(len(frames) - 1):
        T_full = frames[i]
        I_full = frames[i + 1]

        # Crop central 80% of template (as in paper)
        T = crop_template(T_full, ratio=0.8)
        I = crop_template(I_full, ratio=0.8)

        # Run MR-ICIA to get projective homography
        Hp = mr_icia(T, I, levels=args.levels, max_iters=args.max_iters)

        # Recover pose from homography
        R, t, (roll, pitch, yaw) = recover_pose(Hp, K)

        estimated_step_angles.append((roll, pitch, yaw))
        step_pose_results.append({
            "frame":  i,
            "roll":   roll,
            "pitch":  pitch,
            "yaw":    yaw,
            "tx":     float(t[0]),
            "ty":     float(t[1]),
            "tz":     float(t[2]),
        })

        t_global += R_global.T @ t
        R_global = R_global @ R
        roll_global, pitch_global, yaw_global = rotation_to_euler(R_global)
        absolute_pose_results.append({
            "frame":  i,
            "roll":   roll_global,
            "pitch":  pitch_global,
            "yaw":    yaw_global,
            "tx":     float(t_global[0]),
            "ty":     float(t_global[1]),
            "tz":     float(t_global[2]),
        })

        # print(f"Frame {i:04d} -> {i+1:04d} | "
        #       f"roll={roll:7.3f}°  pitch={pitch:7.3f}°  yaw={yaw:7.3f}°")

        print(f"Frame {i:04d} -> {i+1:04d} | "
              f"roll={roll_global:7.3f}°  pitch={pitch_global:7.3f}°  yaw={yaw_global:7.3f}°")

    # --- RMSE evaluation ---
    if gt_poses:
        rmse = compute_rmse_vpair(estimated_step_angles, gt_absolute_angles[1:])
        print("\n--- RMSE vs ground truth ---")
        print(f"  Roll:  {rmse['roll_rmse']:.4f}°")
        print(f"  Pitch: {rmse['pitch_rmse']:.4f}°")
        print(f"  Yaw:   {rmse['yaw_rmse']:.4f}°")
        plot_errors_over_time(
            estimated_step_angles,
            gt_absolute_angles[1:],
            output_path="moving_rmse2.png",
        )

    # --- Save results to CSV ---
    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=absolute_pose_results[0].keys())
        writer.writeheader()
        writer.writerows(absolute_pose_results)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
