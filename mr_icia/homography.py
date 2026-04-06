import cv2
import numpy as np


def build_homography(p: np.ndarray) -> np.ndarray:
    """
    Convert 8-parameter vector to 3x3 homography matrix.
    Parameterisation from eq. (1) in Martinez et al. (2011):

        H = [[1+p0,  p1,  p2],
             [ p3,  1+p4, p5],
             [ p6,   p7,  1 ]]

    Args:
        p: 8-element parameter vector [p0 .. p7].

    Returns:
        3x3 homography matrix as float64.
    """
    return np.array([
        [1.0 + p[0],       p[1],  p[2]],
        [      p[3], 1.0 + p[4],  p[5]],
        [      p[6],       p[7],  1.0 ]
    ], dtype=np.float64)


def warp_image(img: np.ndarray, H: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Warp image I by homography H using inverse mapping.
    This applies W(x; p) to map the current image onto the template domain.

    Args:
        img:   Source image to warp (float32).
        H:     3x3 homography matrix.
        shape: Output shape (height, width) matching the template.

    Returns:
        Warped image as float32.
    """
    h, w = shape
    warped = cv2.warpPerspective(
        img, H, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )
    return warped.astype(np.float32)


def compose_warp(p: np.ndarray, delta_p: np.ndarray) -> np.ndarray:
    """
    ICIA compositional warp update from eq. (5):
        W(x; p) <- W(x; p) o W(x; delta_p)^-1

    In matrix form:
        H_new = H(p) @ inv(H(delta_p))

    Then extract the new 8 parameters from H_new.

    Args:
        p:       Current 8-parameter vector.
        delta_p: Parameter increment from the least-squares solve.

    Returns:
        Updated 8-parameter vector.
    """
    H_p  = build_homography(p)
    H_dp = build_homography(delta_p)

    H_new = H_p @ np.linalg.inv(H_dp)
    H_new /= H_new[2, 2]  # normalise so bottom-right element = 1

    p_new = np.array([
        H_new[0, 0] - 1.0,  # p0
        H_new[0, 1],         # p1
        H_new[0, 2],         # p2
        H_new[1, 0],         # p3
        H_new[1, 1] - 1.0,  # p4
        H_new[1, 2],         # p5
        H_new[2, 0],         # p6
        H_new[2, 1],         # p7
    ])
    return p_new


def propagate_params(p: np.ndarray) -> np.ndarray:
    """
    Scale parameters when moving from a coarser level to the next finer level,
    following eq. (6) from Martinez et al. (2011):

        p_i unchanged  for i in {0, 1, 3, 4}   (rotation / shear)
        p_i *= 2       for i in {2, 5}           (translation)
        p_i /= 2       for i in {6, 7}           (perspective)

    Args:
        p: 8-parameter vector from the current (coarser) pyramid level.

    Returns:
        Scaled 8-parameter vector for the next (finer) pyramid level.
    """
    p_new = p.copy()
    p_new[2] *= 2.0   # x-translation
    p_new[5] *= 2.0   # y-translation
    p_new[6] /= 2.0   # perspective x
    p_new[7] /= 2.0   # perspective y
    return p_new