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


def detect_lp_width(model, I,  MAXWIDTH, net_step, out_size, threshold):
	
	factor = min(1, MAXWIDTH/I.shape[1])
	w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()
	
	w += (w%net_step!=0)*(net_step - w%net_step)
	h += (h%net_step!=0)*(net_step - h%net_step)

	Iresized = cv2.resize(I,(w,h), interpolation = cv2.INTER_CUBIC)
	T = Iresized.copy()

	T = T.reshape((1,T.shape[0],T.shape[1],T.shape[2]))

	start 	= time.time()
	Yr 		= model.predict(T)
	Yr 		= np.squeeze(Yr)
	elapsed = time.time() - start

	L,TLps = reconstruct_new (I, Iresized, Yr, out_size, threshold)

	return L,TLps,elapsed


def reconstruct_new(Iorig, I, Y, out_size, threshold=.9):

	net_stride 	= 2**4 
	side = ((208. + 40.)/3.)/net_stride 

	Probs = Y[...,0]
	Affines = Y[...,-6:] 
	rx,ry = Y.shape[:2]

	xx,yy = np.where(Probs>threshold)
	WH = getWH(I.shape)
	MN = WH/net_stride
	vxx = vyy = 0.5 
	base = lambda vx,vy: np.matrix([[-vx,-vy,1.],[vx,-vy,1.],[vx,vy,1.],[-vx,vy,1.]]).T
	labels = []

	for i in range(len(xx)):
		y,x = xx[i],yy[i]
		affine = Affines[y,x]
		prob = Probs[y,x]