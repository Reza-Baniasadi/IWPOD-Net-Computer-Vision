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
		side  = int(ratio*288.)
		bound_dim = min(side + (side%(2**4)), WPODResolution)
		Llp,LlpImgs,_ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, 2**4, (240,80), lp_threshold)
		plate.append(Llp)
		plateimgs.append(LlpImgs)
	return (plate, plateimgs, vehicles)


def ocr_plates(tfnet_ocr, result,  imgcv, platelist, plateimgslist):
	listocr = [];
	listimgs = [];
	numplates = 0;
	for LlpImgs in plateimgslist:
			if len(LlpImgs):
				Llp = platelist[numplates]
				Ilp = LlpImgs[0]
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
				print(Ilp.shape)
				ptspx = adjust_pts(Llp[0].pts, result[numplates])
				draw_losangle(imgcv, ptspx, (0, 0, 255), 3)
				ocr = tfnet_ocr.return_predict(Ilp * 255.)
				ocr = nms_darkflow(ocr)
				ocr.sort(key=lambda x: x['topleft']['x'])
				lp_str = ''.join([r['label'] for r in ocr])