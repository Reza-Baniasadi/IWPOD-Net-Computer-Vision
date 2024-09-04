import tensorflow as tf
from typing import Tuple


# ---------------- Utility Loss Functions -----------------
def clamp_tensor(tensor: tf.Tensor, epsilon: float = 1e-10) -> tf.Tensor:
    """
    Clamp tensor values to avoid log(0)
    """
    return tf.clip_by_value(tensor, epsilon, 1.0 - epsilon)


def compute_log_prob_loss(true_probs: tf.Tensor, predicted_probs: tf.Tensor, tensor_shape: Tuple[int, int, int, int]) -> tf.Tensor:
    """
    Compute negative log likelihood loss for per-pixel classification
    """
    batch, height, width, channels = tensor_shape
    predicted_probs = clamp_tensor(predicted_probs)
    predicted_probs = -tf.math.log(predicted_probs)
    predicted_probs = predicted_probs * true_probs
    predicted_probs = tf.reshape(predicted_probs, (batch, height * width * channels))
    return tf.reduce_sum(predicted_probs, axis=1)


def compute_l1_error(true_values: tf.Tensor, predicted_values: tf.Tensor, tensor_shape: Tuple[int, int, int, int]) -> tf.Tensor:
    """
    Compute L1 error for regression targets
    """
    batch, height, width, channels = tensor_shape
    diff = tf.reshape(true_values - predicted_values, (batch, height * width * channels))
    diff = tf.abs(diff)
    return tf.reduce_sum(diff, axis=1)


# ---------------- Object Classification Loss -----------------
def compute_object_classification_loss(true_labels: tf.Tensor, predicted_labels: tf.Tensor) -> tf.Tensor:
    """
    Compute object vs background classification loss
    """
    weight_object = 0.6
    weight_background = 0.6
    batch, height, width = tf.shape(true_labels)[0], tf.shape(true_labels)[1], tf.shape(true_labels)[2]

    object_true = true_labels[..., 0]
    object_pred = predicted_labels[..., 0]

    background_true = 1.0 - object_true
    background_pred = 1.0 - object_pred

    loss = weight_object * compute_log_prob_loss(object_true, object_pred, (batch, height, width, 1))
    loss += weight_background * compute_log_prob_loss(background_true, background_pred, (batch, height, width, 1))
    return loss


# ---------------- Plate Localization Loss -----------------
def compute_plate_localization_loss(true_map: tf.Tensor, predicted_map: tf.Tensor) -> tf.Tensor:
    """
    Compute L1 loss for predicted license plate corner points using affine transformation
    """
    batch, height, width = tf.shape(true_map)[0], tf.shape(true_map)[1], tf.shape(true_map)[2]

    object_mask = true_map[..., 0]
    affine_prediction = predicted_map[..., 1:]
    true_points = true_map[..., 1:]

    # Affine decomposition
    affine_x = tf.stack([tf.maximum(affine_prediction[..., 0], 0.0),
                         affine_prediction[..., 1],
                         affine_prediction[..., 2]], axis=3)

    affine_y = tf.stack([affine_prediction[..., 3],
                         tf.maximum(affine_prediction[..., 4], 0.0),
                         affine_prediction[..., 5]], axis=3)

    # Base unit square for license plate corners
    unit_square = tf.constant([[-0.5, -0.5, 1.0, 0.5, -0.5, 1.0, 0.5, 0.5, 1.0, -0.5, 0.5, 1.0]], dtype=tf.float32)
    unit_square = tf.reshape(unit_square, (1, 1, 1, 12))
    unit_square = tf.tile(unit_square, [batch, height, width, 1])

    predicted_points_list = []

    for i in range(0, 12, 3):
        base_slice = unit_square[..., i:i + 3]
        pred_x = tf.reduce_sum(affine_x * base_slice, axis=3)
        pred_y = tf.reduce_sum(affine_y * base_slice, axis=3)
        predicted_points_list.append(tf.stack([pred_x, pred_y], axis=3))

    predicted_points = tf.concat(predicted_points_list, axis=3)

    object_mask_reshaped = tf.reshape(object_mask, (batch, height, width, 1))
    loss = compute_l1_error(true_points * object_mask_reshaped,
                            predicted_points * object_mask_reshaped,
                            (batch, height, width, predicted_points.shape[-1]))
    return loss


# ---------------- Complete WPOD-NET Loss -----------------
def compute_wpodnet_total_loss(true_tensor: tf.Tensor, predicted_tensor: tf.Tensor) -> tf.Tensor:
    """
    Combine plate localization and classification loss with weights
    """
    weight_localization = 0.6
    weight_classification = 0.6

    total_loss = (weight_localization * compute_plate_localization_loss(true_tensor, predicted_tensor) +
                  weight_classification * compute_object_classification_loss(true_tensor, predicted_tensor))
    return total_loss
