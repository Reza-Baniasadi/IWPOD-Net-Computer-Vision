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