import numpy as np
import cv2
from math import sin, cos
from typing import Tuple


# ---------------- Homography & Perspective Utilities -----------------
def compute_homography_matrix(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Compute the 3x3 homography matrix H such that dst_pts ~ H * src_pts
    """
    A = np.zeros((8, 9))
    for i in range(4):
        p_src = src_pts[:, i]
        p_dst = dst_pts[:, i]
        p_src = p_src.T

        A[i*2, 3:6] = -p_dst[2] * p_src
        A[i*2, 6:] = p_dst[1] * p_src
        A[i*2+1, :3] = p_dst[2] * p_src
        A[i*2+1, 6:] = -p_dst[0] * p_src

    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape((3, 3))
    return H


def perturb_points_randomly(points: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """
    Randomly perturb quadrilateral points for augmentation
    """
    signs = np.array([[-1, 1, 1, -1],
                      [-1, -1, 1, 1]])
    perturbed = np.zeros((2, 4))
    edge_lengths = [np.linalg.norm(points[:, i] - points[:, (i + 1) % 4]) for i in range(4)]

    scales = np.array([(edge_lengths[0] + edge_lengths[2]) / 2 * alpha,
                       (edge_lengths[1] + edge_lengths[3]) / 2 * alpha])

    for i in range(4):
        perturbed[:, i] = points[:, i] + np.random.rand(2) * signs[:, i] * scales
    return perturbed


def warp_license_plate(img: np.ndarray, src_pts: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop and warp quadrilateral region from the image to a rectangle
    """
    target_pts = get_rectangle_points(0, 0, output_size[0], output_size[1])
    src_pts_homog = np.vstack((src_pts, np.ones((1, 4))))
    H = compute_homography_matrix(src_pts_homog, target_pts)
    warped_img = cv2.warpPerspective(
        img, H, output_size,
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderValue=0.0
    )
    return warped_img


def get_rectangle_points(tlx: int, tly: int, brx: int, bry: int) -> np.ndarray:
    """
    Return 3x4 homogeneous coordinates of rectangle corners
    """
    return np.array([
        [tlx, brx, brx, tlx],
        [tly, tly, bry, bry],
        [1.0, 1.0, 1.0, 1.0]
    ], dtype=float)


def generate_perspective_transform_matrix(image_wh: Tuple[int, int],
                                          angles: np.ndarray = np.array([0., 0., 0.]),
                                          z_camera: float = 1000.0,
                                          depth_scale: float = 1000.0) -> np.ndarray:
    """
    Compute homography induced by a 3D rotation for synthetic perspective augmentation
    """
    w, h = image_wh
    rad_angles = np.deg2rad(angles)

    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, cos(rad_angles[0]), sin(rad_angles[0])],
                   [0, -sin(rad_angles[0]), cos(rad_angles[0])]])

    Ry = np.array([[cos(rad_angles[1]), 0, -sin(rad_angles[1])],
                   [0, 1, 0],
                   [sin(rad_angles[1]), 0, cos(rad_angles[1])]])

    Rz = np.array([[cos(rad_angles[2]), sin(rad_angles[2]), 0],
                   [-sin(rad_angles[2]), cos(rad_angles[2]), 0],
                   [0, 0, 1]])

    R = Rx @ Ry @ Rz

    # Original rectangle in 3D
    corners_3d = np.array([[0, 0, w, w],
                           [0, h, 0, h],
                           [0, 0, 0, 0]])

    corners_3d_centered = corners_3d - np.array([[w/2], [h/2], [0]])
    corners_rotated = R @ corners_3d_centered
    corners_rotated[2, :] -= z_camera

    corners_homog = np.vstack((corners_rotated, np.ones((1, 4))))

    # Projection
    P = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, -1.0 / depth_scale, 0]])

    projected = P @ corners_homog
    projected /= projected[2, :]
    projected[:2, :] += np.array([[w / 2], [h / 2]])

    return compute_homography_matrix(corners_homog, projected)
