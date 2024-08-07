import tensorflow as tf
from typing import Any


def logloss(Ptrue, Pred, szs, eps=10e-10):
	b,h,w,ch = szs
	Pred = tf.clip_by_value(Pred, eps, 1.0 - eps)
	Pred = -tf.math.log(Pred)
	Pred = Pred*Ptrue
	Pred = tf.reshape(Pred, (b, h*w*ch))
	Pred = tf.reduce_sum(Pred,1)
	return Pred

def l1(true, pred, szs):
	b,h,w,ch = szs
	res = tf.reshape(true-pred, (b,h*w*ch))
	res = tf.abs(res)
	res = tf.reduce_sum(res,2)
	return res


def clas_loss(Ytrue, Ypred):

	wtrue = 0.6 
	wfalse = 0.6
	b = tf.shape(Ytrue)[0]
	h = tf.shape(Ytrue)[1]
	w = tf.shape(Ytrue)[2]

	obj_probs_true = Ytrue[...,0]
	obj_probs_pred = Ypred[...,0]

	non_obj_probs_true = 1. - Ytrue[...,0]
	non_obj_probs_pred = 1 - Ypred[...,0]

	res  = wtrue*logloss(obj_probs_true,obj_probs_pred,(b,h,w,1))
	res  += wfalse*logloss(non_obj_probs_true,non_obj_probs_pred,(b,h,w,1))
	return res