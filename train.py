# -*- coding: utf-8 -*-
# author: K

from common import training_dir, image_size, training_batch_size
from image_preprocessing import inputs
from recognizer import inference, loss, accuracy, optimizer, dropout_prob

import tensorflow as tf

if __name__ == '__main__':

	images, labels = inputs(training_dir, image_size, training_batch_size, True)

	logits = inference(images, 2, image_size)

	l = loss(logits, labels)

	acc = accuracy(logits, labels)

	opt = optimizer(l)

	saver = tf.train.Saver()

	sess = tf.Session()

	sess.run(tf.global_variables_initializer())

	coord = tf.train.Coordinator()

	threads = tf.train.start_queue_runners(sess, coord = coord)

	for i in range(3001):
		sess.run(opt, feed_dict = {dropout_prob: 0.5})

		if i % 10 == 0:
			res = sess.run([l, acc], feed_dict = {dropout_prob: 1.0})
			print "Training Step: %d, loss: %.5f, Training Accuracy: %.5f" % (i, res[0], res[1])
		if i % 500 == 0:	
			saver.save(sess, "./model/model.ckpt")
			print "Model Saved!"

	coord.request_stop()

	coord.join(threads, stop_grace_period_secs = 10)

	sess.close()
