import cv2
import numpy as np
import random
import glob

from src.utils 	import im2single, getWH, hsv_transform
from src.label	import Label
from src.projection_utils import perspective_transform, find_T_matrix, getRectPts

UseBG = True
bgimages = []
dim0 = 208
BGDataset = 'bgimages\\'


if UseBG:
	imglist = glob.glob(BGDataset + '*.jpg')
	for im in imglist:
		img = cv2.imread(im)
		factor = max(1, dim0/min(img.shape[0:2]))
		img = cv2.resize(img, (0,0), fx = factor, fy = factor).astype('float32')/255
		bgimages.append(img)
		
def random_crop(img, width, height):
	or_height = img.shape[0]
	or_width = img.shape[1]
	top = int(np.random.rand(1)*(or_height - height))
	bottom = int(np.random.rand(1)*(or_width - width))
	crop = img[top:(top+height), bottom:(bottom+width),:]
	return crop

def GetCentroid(pts):
	return np.mean(pts, 1)


def ShrinkQuadrilateral(pts, alpha=0.75):

	centroid = GetCentroid(pts)
	temp = centroid + alpha * (pts.T - centroid)
	return temp.T
