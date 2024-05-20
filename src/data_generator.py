import numpy as np

from tensorflow import keras
from src.sampler import augment_sample, labels2output_map

class ALPRDataGenerator(keras.utils.Sequence):
     def __init__(self, data, batch_size=32, dim =  208, stride = 16, shuffle=True, OutputScale = 1.0):
        'Initialization'
        self.dim = dim
        self.stride = stride
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        self.OutputScale = OutputScale
        self.on_epoch_end()

     def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))