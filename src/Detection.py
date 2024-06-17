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
				listocr.append(lp_str)
				listimgs.append(Ilp)
				numplates = numplates + 1;
			# if len(lp_str) < 7:
			# 	cv2.imshow('Orig', imgcv); 
			# if len(lp_str) > 0:
			# 	cv2.imshow('OCR', Ilp); cv2.waitKey(); cv2.destroyAllWindows()
			# else:
			# 	cv2.waitKey(); cv2.destroyAllWindows()
	return listocr, listimgs

def save_print_files(listocr, listimgs, outputdir, rootname):
	for i in range(0, len(listocr)):
		ocr = listocr[i]
		img = listimgs[i]
		if config.SaveTxt:
			with open(outputdir + '%s_str_%d.txt' % (rootname, i + 1),'w') as f:
				f.write(ocr + '\n')
		if config.SaveImages:
			cv2.imwrite(outputdir + rootname +  '_plate_%d' % (i + 1) + '_ocr.png', img*255.)


def run_all(tfnet_yolo, imgcv, wpod_net, lp_threshold, tfnet_ocr, outputdir, rootname):

	result = detect_vechicle(tfnet_yolo, imgcv)
		
	platelist, plateimgslist, result = scan_vehicles(result,  imgcv, wpod_net, lp_threshold)

	listocr, listimgs = ocr_plates(tfnet_ocr, result,  imgcv, platelist, plateimgslist)
	save_print_files(listocr, listimgs, outputdir, rootname)

	return listocr,listimgs




def SwapCharactersLPMercosul(instring):
	#
	#  Format; AAA0A00
	#
	outstring = list(instring);
	if len(instring) == 7:
		for i in range(0,3):
			outstring[i] = imposeLetter(instring[i])
		outstring[3] = imposeDigit(instring[3])
		outstring[4] = imposeLetter(instring[4])
		for i in range(5,7):
			outstring[i] = imposeDigit(instring[i])
	return "".join(outstring)


def SwapCharactersLPBrazilian(instring):
	#
	#  Format AAA0000
	#
	outstring = list(instring);
	if len(instring) == 7:
		for i in range(0,3):
			outstring[i] = imposeLetter(instring[i])
		for i in range(3,8):
			outstring[i] = imposeDigit(instring[i])
	return "".join(outstring)


def SwapCharactersLPChinese(instring):
	#
	#  Format FLAAAAA (A is any), F is a fake chinese character
	#
	
	#
	#  If seven characters are detected, discards the first one
	#
	outstring = list(instring);
	if len(instring) == 7:
		outstring = outstring[1:]
	if len(outstring) == 6:
			outstring[0] = imposeLetter(outstring[0])
	return "".join(outstring)


def imposeLetterString(instring):
	#
	#  Transform  characters into letters
	#
	outstring = list(instring);
	for i in range(0, len(instring)):
		outstring[i] = imposeLetter(instring[i])
	return "".join(outstring)


def imposeLetter(inchar):
	diglist = '0123456789'
	charlist = 'OIZBASETBS'
	outchar = inchar
	if inchar.isdigit():
		ind = diglist.index(inchar)
		outchar = charlist[ind]
	return outchar

def imposeDigit(inchar):
	charlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
	diglist =  '48006661113191080651011017'
	outchar = inchar
	if inchar.isalpha():
		ind = charlist.index(inchar)
		outchar = diglist[ind]
	return outchar


def ClassifyPlate(img, ocr):

	Debug = False
	offset = 4;
	vminy = []
	vmaxy = []
	vminx = []
	vmaxx = []
	vheight = []
	for car in ocr:
		vminy.append(car['topleft']['y']);
		vmaxy.append(car['bottomright']['y']);
		vminx.append(car['topleft']['x']);
		vmaxx.append(car['bottomright']['x']);
		vheight.append(vmaxy[-1] - vminy[-1]);