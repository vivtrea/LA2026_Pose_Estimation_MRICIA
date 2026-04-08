# LA2026_Pose_Estimation_MRICIA

Linear Algebra 2026 course project on Pose Estimation of UAV using Image Alignment (MR-ICIA)

* [Video explanation by Andrii](https://youtu.be/YYFGFgKblvg?si=SOxoTiXM6sDxkr6T)
* [Video explanation by Olena](https://youtu.be/S5_K-FD2lWw?si=SEZoXYkZmepxlrIP)
* [Video explanation by Mykyta part_1](https://youtu.be/I2tpMi9eH-o) + [Video explanation by Mykyta part_2](https://youtu.be/65daS32X2tk)

## Project Overview

This project implements a technique called MR-ICIA (Multi-Resolution Inverse Compositional Image Alignment) to estimate the position and orientation (pose) of a drone (UAV) using only camera images. By analyzing how the scene changes between consecutive images, we can determine how the drone moved between those moments.

The system follows a frame-to-frame incremental pose estimation approach. Given a sequence of images, the goal is to estimate the relative motion between consecutive frames using the MR-ICIA algorithm.

## Dataset and Experimental Setup

For evaluation, we use the publicly available VPAIR dataset provided by the AerVisLoc project. The dataset consists of aerial video sequences captured by an aerial vehicle, along with ground truth camera poses. These sequences provide a realistic benchmark for testing visual pose estimation algorithms under varying conditions such as motion blur, scale changes, and texture variability.

## How It Works

The core idea is based on linear algebra concepts:

- **Homography**: A transformation that describes how points in one image relate to points in another image(when same object viewed from different perspectives)
- **Image Alignment**: Finding the best transformation that aligns one image to another
- **Pose Recovery**: Using the transformation to calculate the actual movement of the drone

### Pipeline Steps

1. **Frame Extraction**
   Video sequences are processed frame by frame. Each pair of consecutive frames is used to estimate relative motion.

2. **Preprocessing**
   Frames are optionally converted to grayscale and smoothed to reduce noise and improve numerical stability of gradient computations.

3. **Gradient Computation**
   Image gradients are computed for each frame. These gradients are essential for constructing the Jacobian matrix used in the ICIA optimization.

4. **Texture Filtering**
   We compute the gradient magnitude of each frame to assess its texture richness. Frames with insufficient texture (e.g., large homogeneous water regions) are filtered out.

5. **MR-ICIA Optimization**
   The multi-resolution inverse compositional image alignment algorithm is applied to estimate the transformation between frames. This produces an estimate of relative motion (translation and rotation).

6. **Error Evaluation**
   The estimated relative motion is compared with ground truth using:
   - Root Mean Square Error (RMSE)
   - Absolute error

## Current Progress

### Completed Components

1. **Homography Operations** (`homography.py`)
   - Building 3×3 transformation matrices from parameters
   - Warping images using perspective transformations
   - Composing transformations for iterative refinement

2. **ICIA Algorithm** (`icia.py`)
   - Computing image gradients and Jacobians
   - Precomputing Hessian matrices for efficiency
   - Performing iterative alignment steps

3. **Pose Recovery** (`pose.py`)
   - Converting projective transformations to 3D rotations
   - Extracting Euler angles (roll, pitch, yaw)
   - Selecting physically valid solutions

4. **Image Pyramids** (`pyramid.py`)
   - Building multi-resolution representations for robust alignment
   - Starting with coarse alignment and refining at higher resolutions

5. **Main Pipeline** (`pipeline.py`)
   - Loading image sequences from disk
   - Processing consecutive image pairs
   - Saving results to CSV files
   - Optional evaluation against ground truth data

### Key Experimental Observation

A significant finding is that the algorithm's performance degrades dramatically in regions with **low texture**, such as water surfaces. This behavior is consistent with the theoretical foundations of the method. Since the Jacobian matrix is constructed from image gradients, regions with little to no texture produce gradients that are close to zero. As a result, the corresponding Hessian matrix becomes ill-conditioned, making the least-squares optimization unstable and leading to large estimation errors.

To investigate this effect, we introduced a simple but effective filtering strategy based on gradient magnitude. Frames that do not contain sufficient texture are excluded from the estimation process. This modification is directly motivated by the linear algebraic properties of the problem, as it improves the conditioning of the system being solved.

### Experimental Results

When all frames are used, including those containing large low-texture regions such as water surfaces, the error exhibits strong instability. In particular, we observe sharp spikes in all three rotation components (roll, pitch, and yaw), with errors exceeding the 5° threshold by a large margin. These spikes are not random but are consistently associated with frames where the scene lacks sufficient texture.

From a theoretical perspective, this behavior is expected. In low-texture regions, image gradients become very small, which leads to near-zero rows in the Jacobian matrix. As a result, the Hessian matrix becomes ill-conditioned, and the least-squares solution becomes highly sensitive to noise. In such cases, the optimization problem is effectively under-constrained, and the estimated motion parameters become unreliable. Consequently, the algorithm produces large, unstable jumps in the estimated pose, which manifest as spikes in the error plots.

After removing frames with insufficient gradient magnitude, the behavior of the system changes significantly. The error becomes much more stable over time, and the large spikes observed previously are almost entirely eliminated. Most of the remaining error values stay close to or below the predefined threshold, indicating a substantial improvement in estimation quality.

This result confirms that the instability observed in the original pipeline is not due to random noise or implementation issues, but rather due to the fundamental limitations of gradient-based methods in low-texture environments. By filtering out such frames, we ensure that the optimization is performed only on well-conditioned data, where the Jacobian contains sufficient information to reliably estimate motion parameters.

### Features Implemented

- Multi-resolution processing for better convergence
- Texture sufficiency checking in enhanced pipeline
- Camera calibration support (known intrinsics or estimation)
- Visualization of alignment errors over time
- Evaluation metrics (RMSE) against ground truth poses

### Usage Example

```bash
# With ground truth evaluation
python mr_icia/pipeline.py --data_dir ./data/vpair_sample/queries --fx 750.626 --fy 750.263 --cx 402.410 --cy 292.988 --output results --pose_file data/vpair_sample/poses_query.txt

# regarding low texture
python mr_icia/pipeline_added_texture.py --data_dir ./data/vpair_sample/queries --fx 750.626 --fy 750.263 --cx 402.410 --cy 292.988 --output results --pose_file data/vpair_sample/poses_query.txt
```

## Technical Details

### Linear Algebra Concepts Used

- **Matrix Operations**: Homography matrices, rotation matrices
- **Vector Calculations**: Parameter vectors for transformations
- **Singular Value Decomposition**: For normalization and decomposition
- **Least Squares Optimization**: For finding optimal transformations
- **Gaussian Elimination**: Implicit in matrix inversion operations
- **Condition Numbers**: Understanding numerical stability of systems

### Dependencies

- Python 3.x
- OpenCV
- NumPy
- Matplotlib (for visualization)

Install with:
```bash
pip install -r requirements.txt
```

## Planned Work

While the current implementation focuses on relative motion estimation, the final objective is to recover absolute camera trajectory (specifically, new absolute coordinates).

The remaining tasks include:

- **Trajectory Integration**: Accumulate relative transformations to reconstruct the global trajectory
- **Initialization with Known Pose**: Use the first frame's ground truth pose as a reference point and propagate transformations forward
- **Drift Analysis**: Evaluate how errors accumulate over time and analyze long-term stability
- **Full Pipeline Evaluation**: Compare reconstructed trajectories with ground truth using:
  - Absolute Trajectory Error (ATE)
  - Relative Pose Error (RPE)

## References

Based on the method described in:
[Martinez et al. (2011) - "Random Sample Matrix Factorization for Distributed Video Analysis"](https://oa.upm.es/13248/1/INVE_MEM_2011_111191.pdf)

VPAIR dataset:
[Schleiss et al. (2022) - VPAIR dataset]
(https://zenodo.org/records/6473989#.YmB_XC8esQw)
