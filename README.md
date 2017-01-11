# tensorflow_cnn
The implementation of tensorflow cnn based on the file queue pipeline. 

# Desc
There are many implementations of convolutional neural networks with tensorflow. However, most of the users learn the cnn from the tensorflow tutorials which provide less information about reading arbitrary size images and applying CNN on them. This repository shows a clean implementation of how tensorflow file queue works and how they could be used on any neural networks with any size of images.

# In addition
You could also learn some more details on how file queue pipeline works from https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10

# Usage
a. Create a directory, such as "data"

b. Create sub-directories such as "dogs", "cats" in "data" directory

c. Place arbitrary sizes of images into corresponding sub-directories

d. Train and test your data

e. Most inportant is: have fun with your network

PS: You could change the neural network architectures whatever you want in "inference" function.
