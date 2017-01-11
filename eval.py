# -*- coding: utf-8 -*-
# author: K

from recognizer import *

def predict(saver, logits):
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state("./")
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


def eval_once(saver, acc):
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state("./")
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print "Model Loaded!"
		else:
			print "Model Not Found!"
			return

		coord = tf.train.Coordinator()
		
		threads = tf.train.start_queue_runners(sess, coord = coord)

		print "%.5f" % sess.run(acc)

		coord.request_stop()

		coord.join(threads, stop_grace_period_secs = 10)


if __name__ == '__main__':
	with tf.Graph().as_default():
		images, labels = inputs("./data", [64, 64], 294, True)

		logits = inference(images, 2, 1.0, True)

		acc = accuracy(logits, labels)

		saver = tf.train.Saver()

		eval_once(saver, acc)

		#predict(saver, logits)
