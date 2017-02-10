# -*- coding: utf-8 -*-
# author: K

from common import testing_dir, testing_batch_size, image_size
from recognizer import inference, accuracy, loss, dropout_prob
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

def eval_average(saver, logits, acc, labels):
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state("./model/")
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print "Model Loaded"

		else:
			print "Model Not Found"
			return

		res_list = []

		coord = tf.train.Coordinator()
	
		threads = tf.train.start_queue_runners(sess, coord = coord)

		for i in range(30):

			l = tf.argmax(labels, 1)
			p = tf.argmax(logits, 1)

			res_list.append(sess.run([acc], feed_dict = {dropout_prob: 1.0}))

		coord.request_stop()

		coord.join(threads, stop_grace_period_secs = 10)

		print np.mean(res_list)



if __name__ == '__main__':

	images, labels = inputs(testing_dir, image_size, testing_batch_size, True)

	logits = inference(images, 2, image_size)

	acc = accuracy(logits, labels)

	saver = tf.train.Saver()

	#predict_image(saver, logits)

	#eval_once(saver, logits, acc, labels)

	#predict(saver, logits)

	eval_average(saver, logits, acc, labels)
