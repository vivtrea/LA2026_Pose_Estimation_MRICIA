import cv2
import numpy as np

from homography import build_homography, warp_image


def compute_jacobian_and_hessian(template: np.ndarray):
    """
    Precompute the per-pixel Jacobian J and the Hessian H = J^T J
    for the ICIA algorithm (eqs. 3-4 in Martinez et al. 2011).

    Because ICIA evaluates the Jacobian at the TEMPLATE gradient (not the
    warped image), H is constant throughout the iteration at this pyramid
    level and only needs to be computed ONCE — this is the key efficiency
    gain over standard Lucas-Kanade.

    The 8-parameter homography Jacobian dW/dp at identity warp (p=0) is:
        row 0 (dx): [x, y, 1, 0, 0, 0, -x^2, -xy]
        row 1 (dy): [0, 0, 0, x, y, 1, -xy, -y^2]

    Args:
        template: Grayscale template image (float32, H x W).

    Returns:
        J_flat:  (H*W, 8) Jacobian matrix (one row per pixel).
        H_mat:   (8, 8)   Precomputed Hessian = J_flat.T @ J_flat.
    """
    h, w = template.shape

    gx = cv2.Sobel(template, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(template, cv2.CV_32F, 0, 1, ksize=3)

    #pixel coordinate grids
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

    #per-pixel Jacobian dW/dp shape (H, W, 2, 8)
    #row 0 = x-component, row 1 = y-component
    zeros = np.zeros_like(xs)
    ones  = np.ones_like(xs)

    dW = np.stack([
        [xs,      ys,   ones, zeros, zeros, zeros, -xs*xs, -xs*ys],
        [zeros, zeros, zeros,    xs,    ys,  ones, -xs*ys, -ys*ys],
    ], axis=0)
    dW = dW.transpose(2, 3, 0, 1)

    grad_T = np.stack([gx, gy], axis=-1)[:, :, np.newaxis, :]

    J = (grad_T @ dW)[:, :, 0, :]

    J_flat = J.reshape(-1, 8)

    H_mat = J_flat.T @ J_flat

    return J_flat, H_mat


def icia_step(
    template:  np.ndarray,
    current:   np.ndarray,
    p:         np.ndarray,
    J_flat:    np.ndarray,
    H_inv:     np.ndarray,
) -> tuple:
    """
    One ICIA iteration: warp current image, compute pixel error,
    solve for parameter increment delta_p (eq. 3).

    delta_p = H^-1 * sum_x( J(x)^T * [I(W(x;p)) - T(x)] )

    Args:
        template: Template image (float32).
        current:  Current image to align to template (float32).
        p:        Current 8-parameter vector.
        J_flat:   Precomputed Jacobian (H*W, 8).
        H_inv:    Precomputed inverse Hessian (8, 8).

    Returns:
        delta_p: 8-element parameter update.
        mse:     Mean squared error between warped image and template
                 (used for termination criterion T2).
    """
    H_mat  = build_homography(p)
    warped = warp_image(current, H_mat, template.shape)

    #pixel-wise error: I(W(x;p)) - T(x), flattened
    error = (warped - template).ravel()

    #least-squares solution for delta_p
    delta_p = H_inv @ (J_flat.T @ error)

    mse = float(np.mean(error ** 2))
    return delta_p, mse