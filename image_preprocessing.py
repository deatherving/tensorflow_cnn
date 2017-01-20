# -*- coding: utf-8 -*-
# author: K

from common import *
from os import walk
from os.path import join
import numpy as np

num_examples_per_epoch = 25000

def _match_images_and_labels(image, label, min_queue_example, batch_size, shuffle, one_hot):
	num_preprocess_threads = 16

	if shuffle:
		images, label_batch = tf.train.shuffle_batch([image, label], batch_size = batch_size, num_threads = num_preprocess_threads, capacity = min_queue_example + 3 * batch_size, min_after_dequeue=min_queue_example)
	else:
		images, label_batch = tf.train.batch([image, label], batch_size = batch_size, num_threads = num_preprocess_threads, capacity = min_queue_example + 3 * batch_size)

	if one_hot:
		return images, label_batch
	else:
		return images, tf.reshape(label_batch, [batch_size])

def _read_image(file_name_queue):
	label = file_name_queue[1]
	content = tf.read_file(file_name_queue[0])
	image = tf.image.decode_jpeg(content, channels = 3)
	return image, label


def _one_hot_encoding(key, classes):
	res = [0] * len(classes)
	idx = classes[key]
	res[idx] = 1
	return res


def inputs(path, output_size, batch_size, one_hot = True):
	filenames = []
	labels = []
	classes = {}
	class_idx = 0

	for root, dirs, files in walk(path):
		if len(dirs) != 0:
			for d in dirs:
				classes[d] = class_idx
				class_idx += 1
		for f in files:
			path = join(root, f)
			for key in classes:
				if key in path:
					if one_hot:
						labels.append(_one_hot_encoding(key, classes))
					else:
						labels.append(classes[key])
			filenames.append(path)

	tensor_labels = tf.pack(labels)

	file_name_list = tf.train.slice_input_producer([filenames, tensor_labels])

	read_input, label = _read_image(file_name_list)

	reshaped_image = tf.cast(read_input, tf.float32)

	output_height = output_size[0]
	output_width = output_size[1]

	resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, output_width, output_height)

	float_image = tf.image.per_image_standardization(resized_image)

	min_fraction_of_examples_in_queue = 0.4

	min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

	return _match_images_and_labels(float_image, label, min_queue_examples, batch_size, False, one_hot)



if __name__ == '__main__':
	data = inputs("./data", [64, 64], 128, False)
