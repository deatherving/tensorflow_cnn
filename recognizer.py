# -*- coding: utf-8 -*-
# author: K
# desc: 1. basic cnn nets
#	2. vgg19


from common import *

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
	# pooling layer
	if mode == "MAX":
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def dropout(h, drop_prob):
	return tf.nn.drop_out(h, drop_prob)

def inference(images, num_classes, dropout_prob):
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

	'''
	# third layer
	W_conv3 = weights_intializer([5, 5, 128, 64])
	b_conv3 = bias_intiializer([64])

	h_conv3 = tf.nn.relu(conv_layer(h_poo2, W_conv3) + b_conv3)
	h_pool3 = pooling_layer(h_conv3)

	'''

	# fully connect layer
	# The shape of the W_fc1 is based on the max_pooling times, normally you can compute the shape dynamically.
	W_fc1 = weights_initializer([16 * 16 * 64, 1024])
	b_fc1 = bias_initializer([1024])
	
	h_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

	h_fc_drop1 = tf.nn.dropout(h_fc1, dropout_prob)

	# second fully connected

	W_fc2 = weights_initializer([1024, 256])
	b_fc2 = bias_initializer([256])
	
	h_fc2 = tf.matmul(h_fc_drop1, W_fc2) + b_fc2

	# output
	W_out = weights_initializer([256, num_classes])
	b_out = bias_initializer([num_classes])

	h_out = tf.matmul(h_fc2, W_out) + b_out

	return h_out

def loss(out, labels):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, labels))
	return loss

def optimizer(loss, learning_rate = 0.0001):
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	return optimizer


def accuracy(out, labels):
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(labels, 1)), tf.float32))
	return accuracy
