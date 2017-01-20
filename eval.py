# -*- coding: utf-8 -*-
# author: K

from common import testing_dir, testing_batch_size, image_size
from recognizer import inference, accuracy, loss
from image_preprocessing import inputs
from PIL import Image
import numpy as np
import tensorflow as tf

def predict_image(sess, logits):
	print sess.run(tf.argmax(logits, 1))

def predict(saver, logits):
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state("./model/")
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print "Model Loaded!"
		else:	
			print "Model Not Found"
			return

		coord = tf.train.Coordinator()
		
		threads = tf.train.start_queue_runners(sess, coord = coord)

		print sess.run(tf.argmax(logits, 1))

		coord.request_stop()

		coord.join(threads, stop_grace_period_secs = 10)


def eval_once(saver, logits, acc, labels):
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state("./model/")
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print "Model Loaded!"
		else:
			print "Model Not Found!"
			return

		coord = tf.train.Coordinator()
		
		threads = tf.train.start_queue_runners(sess, coord = coord)

		l = tf.argmax(labels,1)
		p = tf.argmax(logits,1)

		print sess.run([l,p, tf.equal(l,p), acc])

		coord.request_stop()

		coord.join(threads, stop_grace_period_secs = 10)


if __name__ == '__main__':
	with tf.Graph().as_default():

		images, labels = inputs(testing_dir, image_size, testing_batch_size, True)

		logits = inference(images, 2, image_size, 1.0)

		acc = accuracy(logits, labels)

		saver = tf.train.Saver()

		#predict_image(saver, logits)

		eval_once(saver, logits, acc, labels)

		#predict(saver, logits)
