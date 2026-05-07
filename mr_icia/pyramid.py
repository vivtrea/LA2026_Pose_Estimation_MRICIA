import cv2
import numpy as np


def build_pyramid(image: np.ndarray, levels: int) -> list:
    """
    Build a Gaussian image pyramid by downsampling the image by factor 2
    at each level, as described in the MR-ICIA paper (Section II-B).

    Args:
        image:  Grayscale image as float32 numpy array (H, W).
        levels: Number of pyramid levels.

    Returns:
        List of images from coarsest (index 0) to finest (index levels-1).
        Index 0 is the lowest resolution — this is where the MR loop starts.
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    pyramid = [image]
    for _ in range(levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))

    # reverse so index 0 = coarsest (lowest resolution)
    pyramid.reverse()
    return pyramid