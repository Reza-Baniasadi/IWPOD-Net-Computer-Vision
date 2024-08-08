
import numpy as np
import cv2
from math import sin, cos


def find_T_matrix(pts,t_pts):
	A = np.zeros((8,9))
	for i in range(0,4):
		xi  = pts[:,i];
		xil = t_pts[:,i];
		xi  = xi.T
		
		A[i*2,   3:6] = -xil[2]*xi
		A[i*2,   6: ] =  xil[1]*xi
		A[i*2+1,  :3] =  xil[2]*xi
		A[i*2+1, 6: ] = -xil[0]*xi

	
	[U,S,V] = np.linalg.svd(A)
	H = V[-1,:].reshape((3,3))

	return H