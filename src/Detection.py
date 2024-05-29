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

def scan_vehicles(vehicles,  imgcv, wpod_net, lp_threshold):
	plate = []
	plateimgs = []
	if len(vehicles) == 0:
		vehicles = [{'label': 'car',  'confidence': 1,  'topleft': {'x': 1, 'y': 1}, 'bottomright': {'x': imgcv.shape[1], 'y': imgcv.shape[0]}}]
	for car in vehicles:
		tlx = car['topleft']['x'];
		tly = car['topleft']['y'];
		brx = car['bottomright']['x'];
		bry = car['bottomright']['y'];
		Ivehicle = imgcv[tly:bry, tlx:brx]

		WPODResolution = 416 
		ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])

	