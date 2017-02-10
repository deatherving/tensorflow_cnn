# -*- coding: utf-8 -*-
# author: K
# desc: 1. basic cnn nets
#	2. vgg19

import tensorflow as tf

pooling_times = 0

dropout_prob = tf.placeholder(tf.float32)


def weights_initializer(shape):
	values = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(values)

def bias_initializer(shape):
	values = tf.constant(0.1, shape = shape)
	return tf.Variable(values)

def conv_layer(x, W):
	# convolutional layer
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

def pooling_layer(x, mode = "MAX"):
	global pooling_times
	# pooling layer
	if mode == "MAX":
		pooling_times += 1
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def dropout(h, drop_prob):
	return tf.nn.drop_out(h, drop_prob)

def inference(images, num_classes, image_size):
	global pooling_times
	global dropout_prob

	# first layer
	W_conv1 = weights_initializer([5, 5, 3, 64])
	b_conv1 = bias_initializer([64])

	h_conv1 = tf.nn.relu(conv_layer(images, W_conv1) + b_conv1)
	h_pool1 = pooling_layer(h_conv1)

	# second layer
	W_conv2 = weights_initializer([5, 5, 64, 64])
	b_conv2 = bias_initializer([64])

	h_conv2 = tf.nn.relu(conv_layer(h_pool1, W_conv2) + b_conv2)
	h_pool2 = pooling_layer(h_conv2)

	# third layer
	W_conv3 = weights_initializer([5, 5, 64, 128])
	b_conv3 = bias_initializer([128])

	h_conv3 = tf.nn.relu(conv_layer(h_pool2, W_conv3) + b_conv3)
	
	# forth layer
	W_conv4 = weights_initializer([5, 5, 128, 64])
	b_conv4 = bias_initializer([64])

	h_conv4 = tf.nn.relu(conv_layer(h_conv3, W_conv4) + b_conv4)
	h_pool4 = pooling_layer(h_conv4)

	# fully connect layer
	# The shape of the W_fc1 is based on the max_pooling times, normally you can compute the shape dynamically.
	
	flat_height = image_size[0] / 2**pooling_times
	flat_width = image_size[1] / 2**pooling_times

	W_fc1 = weights_initializer([flat_height * flat_width * 64, 2048])
	b_fc1 = bias_initializer([2048])
	
	h_flat = tf.reshape(h_pool4, [-1, flat_height * flat_width * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

	h_fc_drop1 = tf.nn.dropout(h_fc1, dropout_prob)

	# second fully connected

	W_fc2 = weights_initializer([2048, 256])
	b_fc2 = bias_initializer([256])

	h_fc2 = tf.nn.relu(tf.matmul(h_fc_drop1, W_fc2) + b_fc2)

	# output
	W_out = weights_initializer([256, num_classes])
	b_out = bias_initializer([num_classes])

	h_out = tf.matmul(h_fc2, W_out) + b_out

	return h_out

def loss(out, labels):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, labels))
	return loss

def optimizer(loss, learning_rate = 0.0001):
	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
	return optimizer

def predictions(out):
	preds = tf.argmax(out, 1)
	return preds

def accuracy(out, labels):
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(labels, 1)), tf.float32))
	return accuracy

