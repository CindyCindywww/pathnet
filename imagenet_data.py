from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import input_data
import pathnet
import os
import numpy as np
import time
import random
from PIL import Image
import scipy.misc as misc

def create_file_queue(dir_path):
	filename = [] 
	labels = []
	for i in range(10):
		filename.extend(os.listdir(dir_path+"/"+str(i)))
		_len = len(os.listdir(dir_path+'/' + str(i)))
		labels.extend([i for j in range(_len)])
	print("sample number : "+str(len(filename)))
	filename_queue = tf.train.slice_input_producer([filename,labels], shuffle=True)
	return filename_queue[0], filename_queue[1]

def onehot(index, length):
	""" It creates a one-hot vector with a 1.0 in
		position represented by index 
	"""
	onehot = np.zeros(length)
	onehot[index] = 1.0
	return onehot


def read_batch(sess, filenames, labels, batch_size, images_source):
	batch_images = []
	batch_labels = []
	#print(images_source)
	for i in range(batch_size):
		filename, label = sess.run([filenames, labels])
		batch_images.append(preprocess_image(images_source+"/"+str(label)+"/"+filename))
		batch_labels.append(onehot(label, 10))
		#print(label)
	return batch_images, batch_labels

def preprocess_image(image_path):
	""" It reads an image, it resize it to have the lowest dimesnion of 256px,
		it randomly choose a 224x224 crop inside the resized image and normilize the numpy 
		array subtracting the ImageNet training set mean

		Args:
			images_path: path of the image

		Returns:
			cropped_im_array: the numpy array of the image normalized [width, height, channels]
	"""
	IMAGENET_MEAN = [123.68, 116.779, 103.939] # rgb format

	img = Image.open(image_path).convert('RGB')

	# resize of the image (setting lowest dimension to 256px)
	if img.size[0] < img.size[1]:
		h = int(float(256 * img.size[1]) / img.size[0])
		img = img.resize((256, h), Image.ANTIALIAS)
	else:
		w = int(float(256 * img.size[0]) / img.size[1])
		img = img.resize((w, 256), Image.ANTIALIAS)

	# random 224x224 patch
	x = random.randint(0, img.size[0] - 224)
	y = random.randint(0, img.size[1] - 224)
	img_cropped = img.crop((x, y, x + 224, y + 224))

	cropped_im_array = np.array(img_cropped, dtype=np.float32)

	for i in range(3):
		cropped_im_array[:,:,i] -= IMAGENET_MEAN[i]
	cropped_im_array = cropped_im_array / 255.0
	cropped_im_array_shape = np.shape(cropped_im_array)
	cropped_im_array = np.resize(cropped_im_array,(cropped_im_array_shape[0]*cropped_im_array_shape[1]*cropped_im_array_shape[2]))
	#for i in range(3):
	#	mean = np.mean(img_c1_np[:,:,i])
	#	stddev = np.std(img_c1_np[:,:,i])
	#	img_c1_np[:,:,i] -= mean
	#	img_c1_np[:,:,i] /= stddev

	return cropped_im_array
	
'''
#return the numble of images in a folder
def imge_size(im_path):
	n = 0
	for f in os.listdir(im_path):
		n += 1
	return n
'''

if __name__ == "__main__":
	data_folder_task1 = '../data_set/imagenet/task1'
	data_folder_task2 = '../data_set/imagenet/task1'
	filename, label = create_file_queue(data_folder_task1)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess,coord=coord)
	for i in range(50):
		batch_images, batch_labels = read_batch(sess, filename, label, 16, data_folder_task1)
	#print(np.array(batch_images).shape)
	#print(batch_labels[0])
	#print(batch_images[0])

	coord.request_stop()
	coord.join(threads)

	

	
