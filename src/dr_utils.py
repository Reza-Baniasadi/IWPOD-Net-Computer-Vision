import numpy as np
import cv2


# ---------------- Draw Bounding Box -----------------
def draw_bounding_box(image, label, color=(255, 0, 0), thickness=1):
    """
    Draw a rectangular bounding box based on normalized coordinates from a label object
    """
    img_wh = np.array(image.shape[1::-1]).astype(int)  # width, height
    top_left = tuple((label.tl() * img_wh).astype(int).tolist())
    bottom_right = tuple((label.br() * img_wh).astype(int).tolist())
    cv2.rectangle(image, top_left, bottom_right, color, thickness=thickness)


# ---------------- Draw Quadrilateral -----------------
def draw_quadrilateral(image, points, color=(1.0, 1.0, 1.0), thickness=1):
    """
    Draw a quadrilateral given a 2x4 array of points (x, y)
    """
    assert points.shape == (2, 4), "Points array must be of shape 2x4"

    for i in range(4):
        pt1 = tuple(points[:, i].astype(int).tolist())
        pt2 = tuple(points[:, (i + 1) % 4].astype(int).tolist())
        cv2.line(image, pt1, pt2, color, thickness)


# ---------------- Draw Text on Image -----------------
def draw_text_on_image(image, label, text, text_color=(0, 0, 0), bg_color=(255, 255, 255), font_scale=1):
    """
    Draw a text string above a bounding box on the image.
    """
    img_wh = np.array(image.shape[1::-1])  # width, height
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size, _ = cv2.getTextSize(text, font, font_scale, 4)
    bl_corner = label.tl() * img_wh  # bottom-left corner of text

    tl_corner = np.array([bl_corner[0], bl_corner[1] - text_size[1]]) / img_wh
    br_corner = np.array([bl_corner[0] + text_size[0], bl_corner[1]]) / img_wh
    bl_corner /= img_wh

    # Adjust if text goes out of image bounds
    if (tl_corner < 0).any():
        delta = 0. - np.minimum(tl_corner, 0.)
    elif (br_corner > 1).any():
        delta = 1. - np.maximum(br_corner, 1.)
    else:
        delta = 0.

    tl_corner += delta
    br_corner += delta
    bl_corner += delta

    to_tuple = lambda x: tuple((x * img_wh).astype(int).tolist())

    cv2.rectangle(image, to_tuple(tl_corner), to_tuple(br_corner), bg_color, -1)
    cv2.putText(image, text, to_tuple(bl_corner), font, font_scale, text_color, 3)
