import numpy as np

def custom_svd_3x3(A: np.ndarray, max_iters: int = 1000, tol: float = 1e-12) -> tuple:
    """
    Simple SVD for 3x3 matrices via power iteration and deflation.

    For each singular value:
        1. Power iterate on A^T @ A to find the dominant eigenvector v
        2. Compute u = A @ v / sigma
        3. Deflate: A <- A - sigma * u @ v^T
        4. Repeat for next singular value

    Args:
        A:         3x3 input matrix.
        max_iters: Maximum power iterations per singular value.
        tol:       Convergence threshold.

    Returns:
        U:  3x3 left singular vectors.
        S:  3 singular values in descending order.
        Vt: 3x3 right singular vectors transposed.
    """
    A = A.astype(np.float64).copy()
    n = 3

    U_cols = []
    S_vals = []
    V_cols = []

    M = A.copy()

    for _ in range(n):
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)

        for _ in range(max_iters):
            v_new = M.T @ (M @ v)
            norm = np.linalg.norm(v_new)
            if norm < 1e-14:
                break
            v_new = v_new / norm

            if np.linalg.norm(v_new - v) < tol:
                break
            v = v_new

        v = v_new

        Mv = M @ v
        sigma = np.linalg.norm(Mv)

        if sigma < 1e-12:
            #Degenerate: pick any unit vector orthogonal to existing U cols
            u = np.zeros(n)
            u[len(U_cols)] = 1.0
        else:
            u = Mv / sigma

        U_cols.append(u)
        S_vals.append(sigma)
        V_cols.append(v)

        M = M - sigma * np.outer(u, v)

    U  = np.column_stack(U_cols)
    S  = np.array(S_vals)
    Vt = np.vstack(V_cols)

    return U, S, Vt
