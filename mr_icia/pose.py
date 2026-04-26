import cv2
import numpy as np


def recover_pose(Hp: np.ndarray, K: np.ndarray) -> tuple:
    """
    Recover rotation matrix R, translation vector t, and Euler angles
    from the projective homography Hp, using the camera intrinsic matrix K.

    Implements eqs. (13)-(15) from Martinez et al. (2011):

        1. Compute Euclidean homography:  HL = K^-1 @ Hp @ K
        2. Normalise by median singular value (removes scale ambiguity)
        3. Decompose He into (R, t) — 4 candidate solutions
        4. Select correct solution via positive-depth + ground-plane constraint
        5. Extract Euler angles (roll, pitch, yaw) from R

    Args:
        Hp: 3x3 projective homography from the MR-ICIA algorithm.
        K:  3x3 camera intrinsic matrix.

    Returns:
        R:      3x3 rotation matrix.
        t:      3x1 translation vector (scaled by distance to plane).
        angles: Tuple (roll, pitch, yaw) in degrees.
    """
    K_inv = np.linalg.inv(K)

    # Step 1: Euclidean homography (eq. 13)
    HL = K_inv @ Hp @ K

    # Step 2: Normalise by median singular value to remove scale (eq. 13)
    _, sv, _ = np.linalg.svd(HL)
    gamma = np.median(sv)
    # TODO
    if abs(gamma) < 1e-8:
        gamma = 1.0  # guard against degenerate case
    He = HL / gamma

    # Step 3: Decompose He — OpenCV returns 4 solutions
    # We pass identity as K here because He is already in Euclidean space
    num_solutions, Rs, Ts, Ns = cv2.decomposeHomographyMat(He, np.eye(3))

    # Step 4: Select the physically correct solution
    R, t = _select_solution(Rs, Ts, Ns, num_solutions)

    # Step 5: Euler angles from R (eq. 15)
    roll, pitch, yaw = rotation_to_euler(R)

    return R, t, (roll, pitch, yaw)


def _select_solution(
    Rs: list,
    Ts: list,
    Ns: list,
    num_solutions: int,
) -> tuple:
    """
    Pick the physically valid solution from the 4 homography decompositions.

    Criteria (from Section III of the paper):
      1. Positive depth: the scene must be in front of the camera (t[2] > 0).
      2. Ground plane normal closest to [0, 0, 1]^T (flat ground assumption).

    Args:
        Rs, Ts, Ns:    Lists of rotation matrices, translations, and normals.
        num_solutions: Number of valid solutions returned by OpenCV.

    Returns:
        (R, t) of the best solution.
    """
    ground_normal = np.array([0.0, 0.0, 1.0])
    best_idx   = 0
    best_score = -np.inf

    for i in range(num_solutions):
        n = np.array(Ns[i]).ravel()
        t = np.array(Ts[i]).ravel()

        # Positive depth constraint
        if t[2] <= 0:
            continue

        # Alignment with ground normal [0, 0, 1]
        score = float(np.dot(n, ground_normal))
        if score > best_score:
            best_score = score
            best_idx   = i

    return Rs[best_idx], np.array(Ts[best_idx]).ravel()


def rotation_to_euler(R: np.ndarray) -> tuple:
    """
    Extract roll, pitch, yaw (in degrees) from a rotation matrix.
    Convention follows eq. (15) in Martinez et al. (2011).

    Args:
        R: 3x3 rotation matrix.

    Returns:
        (roll, pitch, yaw) in degrees.
    """
    pitch = np.degrees(np.arctan2(-R[2, 0],
                                   np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw

def euler_to_rotation(roll: int, pitch: int, yaw: int) -> np.ndarray:
    rx = np.radians(roll)
    ry = np.radians(pitch)
    rz = np.radians(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ])

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0,           1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1]
    ])

    return Rz @ Ry @ Rx  # ZYX convention