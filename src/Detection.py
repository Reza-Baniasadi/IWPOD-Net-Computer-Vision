import numpy as np
import cv2
from src.keras_utils 	import detect_lp
from src.utils 			import im2single, nms_darkflow, nms_darkflow_target, adjust_pts, print_digits
from src.drawing_utils	import draw_losangle


def detect_vechicle(tfnet_yolo, imgcv):
	result = tfnet_yolo.return_predict(imgcv)
	vehicles = []
	for det in result:
		if (det['label'] in ['car','bus']):
			vehicles.append(det)
	return vehicles