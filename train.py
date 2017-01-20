# -*- coding: utf-8 -*-
# author: K



from recognizer import *


if __name__ == '__main__':
	images, labels = inputs("./data", [64, 64], 128, True)

	logits = inference(images, 2, 0.5)

	l = loss(logits, labels)

	acc = accuracy(logits, labels)

	opt = optimizer(l)

	saver = tf.train.Saver()

	sess = tf.Session()

	sess.run(tf.global_variables_initializer())

	coord = tf.train.Coordinator()

	threads = tf.train.start_queue_runners(sess, coord = coord)

	for i in range(2000):
		sess.run(opt)
		print "Training Step: %d, loss: %.5f, Training Accuracy: %.5f" % (i, sess.run(l), sess.run(acc))

		if i % 1000 == 0:
			saver.save(sess, "./model/model.ckpt")
			print "Model Saved!"

	coord.request_stop()

	coord.join(threads, stop_grace_period_secs = 10)

	sess.close()
