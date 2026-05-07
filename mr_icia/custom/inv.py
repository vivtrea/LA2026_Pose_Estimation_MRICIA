import numpy as np

def inv_8x8(A: np.ndarray) -> np.ndarray:
    """
    Invert an 8x8 matrix via Gauss-Jordan elimination with partial pivoting.

    Augments [A | I] and row-reduces to [I | A^-1].
    Partial pivoting swaps rows to place the largest element on the diagonal
    at each step, which avoids dividing by near-zero values and improves
    numerical stability.

    Args:
        A: 8x8 matrix to invert.

    Returns:
        A_inv: 8x8 inverse matrix.

    Raises:
        ValueError: If matrix is singular.
    """
    n = A.shape[0]

    aug = np.zeros((n, 2 * n), dtype=np.float64)
    aug[:, :n] = A.astype(np.float64)
    aug[:, n:] = np.eye(n)

    for col in range(n):

        max_row = col + np.argmax(np.abs(aug[col:, col]))
        if max_row != col:
            aug[[col, max_row]] = aug[[max_row, col]]

        pivot = aug[col, col]
        if abs(pivot) < 1e-12:
            raise ValueError(f"Matrix is singular at column {col} "
                             f"(pivot={pivot:.2e}).")

        aug[col] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row, col]
            aug[row] -= factor * aug[col]

    return aug[:, n:]

def inv_3x3(H: np.ndarray) -> np.ndarray:
    """
    Compute the exact inverse of a 3x3 matrix analytically.

    Uses the formula: H^-1 = adj(H) / det(H)
    where adj(H) is the adjugate (transpose of cofactor matrix).

    Each cofactor C_ij is the (i,j) minor determinant times (-1)^(i+j).
    For a 3x3 this gives 9 explicit scalar formulas — no numerical
    solver or decomposition required.

    Args:
        H: 3x3 matrix to invert.

    Returns:
        H_inv: 3x3 inverse matrix.

    Raises:
        ValueError: If matrix is singular (det ~ 0).
    """

    a, b, c = H[0, 0], H[0, 1], H[0, 2]
    d, e, f = H[1, 0], H[1, 1], H[1, 2]
    g, h, k = H[2, 0], H[2, 1], H[2, 2]

    C00 = e*k - f*h
    C01 = -(d*k - f*g)
    C02 = d*h - e*g

    C10 = -(b*k - c*h)
    C11 = a*k - c*g
    C12 = -(a*h - b*g)

    C20 = b*f - c*e
    C21 = -(a*f - c*d)
    C22 = a*e - b*d

    det = a*C00 + b*C01 + c*C02

    if abs(det) < 1e-10:
        raise ValueError(f"Homography matrix is singular (det={det:.2e}), "
                          "cannot invert.")

    inv = np.array([
        [C00, C10, C20],
        [C01, C11, C21],
        [C02, C12, C22],
    ], dtype=np.float64) / det

    return inv
