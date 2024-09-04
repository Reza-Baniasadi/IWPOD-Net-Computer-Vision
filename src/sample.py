import cv2
import numpy as np
import random
import glob
from typing import List, Tuple

from src.utils import im2single, getWH, hsv_transform
from src.label import BoundingBox
from src.projection_utils import perspective_transform, find_T_matrix, getRectPts


# ---------------- Background Loader -----------------
class BackgroundManager:
    """
    Load and manage random background images for augmentation
    """
    def __init__(self, use_background: bool = True, bg_folder: str = 'bgimages\\', target_dim: int = 208):
        self.use_background = use_background
        self.bg_folder = bg_folder
        self.target_dim = target_dim
        self.backgrounds: List[np.ndarray] = []

        if self.use_background:
            self._load_backgrounds()

    def _load_backgrounds(self):
        img_paths = glob.glob(self.bg_folder + '*.jpg')
        for path in img_paths:
            img = cv2.imread(path).astype('float32') / 255.0
            scale_factor = max(1, self.target_dim / min(img.shape[:2]))
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            self.backgrounds.append(img)

    def get_random_background(self) -> np.ndarray:
        if not self.backgrounds:
            raise ValueError("No backgrounds loaded")
        return random.choice(self.backgrounds)


# ---------------- Image Cropping -----------------
def crop_random_region(image: np.ndarray, crop_width: int, crop_height: int) -> np.ndarray:
    """
    Randomly crop a sub-region from an image
    """
    max_top = image.shape[0] - crop_height
    max_left = image.shape[1] - crop_width
    top = int(np.random.rand() * max_top)
    left = int(np.random.rand() * max_left)
    return image[top:top + crop_height, left:left + crop_width, :]


# ---------------- Geometry Utilities -----------------
def compute_centroid(points: np.ndarray) -> np.ndarray:
    """
    Compute centroid of quadrilateral points
    """
    return np.mean(points, axis=1)


def shrink_quadrilateral(points: np.ndarray, shrink_factor: float = 0.75) -> np.ndarray:
    """
    Shrink quadrilateral toward its centroid
    """
    centroid = compute_centroid(points)
    shrunk = centroid + shrink_factor * (points.T - centroid)
    return shrunk.T


def compute_polygon_edges(points: np.ndarray) -> List[np.ndarray]:
    """
    Compute homogeneous line equations for edges of a quadrilateral
    """
    edges = []
    for i in range(4):
        p1 = np.hstack((points[:, i], 1))
        p2 = np.hstack((points[:, (i + 1) % 4], 1))
        edges.append(np.cross(p1, p2))
    return edges
