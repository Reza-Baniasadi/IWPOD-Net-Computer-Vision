import sys

import numpy as np
import cv2
import time

from os.path import splitext

from src.label import Label
from src.utils import getWH, nms
from src.projection_utils import getRectPts, find_T_matrix


class DLabel (Label):

	def __init__(self,cl,pts,prob):
		self.pts = pts
		tl = np.amin(pts,1)
		br = np.amax(pts,1)
		Label.__init__(self,cl,tl,br,prob)


def load_model(path,custom_objects={},verbose=0):
	from tensorflow.keras.models import model_from_json

	path = splitext(path)[0]
	with open('%s.json' % path,'r') as json_file:
		model_json = json_file.read()
	model = model_from_json(model_json, custom_objects=custom_objects)
	model.load_weights('%s.h5' % path)
	if verbose: print('Loaded from %s' % path)
	return model