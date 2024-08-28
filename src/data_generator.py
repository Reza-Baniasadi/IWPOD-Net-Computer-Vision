import numpy as np
from tensorflow import keras
from src.sampler import augment_sample, labels2output_map

class LicensePlateBatchGenerator(keras.utils.Sequence):
    """
    Custom data generator for ALPR.
    Applies augmentation and maps labels to output heatmaps.
    """

    def __init__(self, samples, batch_count=32, img_dimension=208, stride=16, shuffle_data=True, scale_factor=1.0):
        """
        Initialization
        """
        self.samples = samples
        self.batch_count = batch_count
        self.img_dimension = img_dimension
        self.stride = stride
        self.shuffle_data = shuffle_data
        self.scale_factor = scale_factor
        self._prepare_index_map()

    def __len__(self):
        """Total number of batches"""
        return int(np.ceil(len(self.samples) / self.batch_count))

    def get_batch(self, batch_number):
        """Public method to get a batch"""
        selected_indexes = self.index_map[batch_number * self.batch_count:(batch_number + 1) * self.batch_count]
        imgs, targets = self._assemble_batch(selected_indexes)
        return imgs, targets

    def shuffle_indexes(self):
        """Shuffle the indexes, call at epoch end"""
        self._prepare_index_map()

    # ---------- Private methods ----------
    def _prepare_index_map(self):
        """Prepare and shuffle index map for batching"""
        self.index_map = list(np.arange(len(self.samples)))

        remainder = self.batch_count - len(self.samples) % self.batch_count
        if remainder != self.batch_count:
            self.index_map += list(np.random.choice(self.index_map, remainder))

        if self.shuffle_data:
            np.random.shuffle(self.index_map)

    def _assemble_batch(self, indexes):
        """Create batch arrays of images and heatmaps"""
        imgs = np.empty((self.batch_count, self.img_dimension, self.img_dimension, 3), dtype=np.float32)
        heatmaps = np.empty((self.batch_count,
                             self.img_dimension // self.stride,
                             self.img_dimension // self.stride,
                             9), dtype=np.float32)

        for i, idx in enumerate(indexes):
            augmented_img, label_points, points_list = augment_sample(
                self.samples[idx][0], self.samples[idx][1], self.img_dimension
            )
            heatmap = labels2output_map(label_points, points_list,
                                        self.img_dimension, self.stride, alfa=0.7)
            imgs[i] = augmented_img * self.scale_factor
            heatmaps[i] = heatmap

        return imgs, heatmaps
