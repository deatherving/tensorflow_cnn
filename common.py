# -*- coding: utf-8 -*-
# author: K

# I think we should define some of the arguments here for convenience.

# The images size we want, if you change the size here, you have to change the architecture in the funtion INFERENCE in file recognizer.py. I will update later to make network automatically adapt the image_size.
image_size = [160, 160]

# training batch size
training_batch_size = 128

# test batch size
testing_batch_size = 200

# training directory
training_dir = "./data"

# testing directory
testing_dir = "./eval_data"

# image input arguments
num_examples_per_epoch = 36000 # controls the capacity of the tf.train.shuffle
min_fraction_of_examples_in_queue = 0.4
is_shuffle = True
