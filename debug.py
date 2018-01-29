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


#return the numble of images in a folder
def imge_size(im_path):
	n = 0
	for f in os.listdir(im_path):
		n += 1
	return n

def onehot(index):
	""" It creates a one-hot vector with a 1.0 in
		position represented by index 
	"""
	onehot = np.zeros(2)
	onehot[index] = 1.0
	return onehot


def read_batch(batch_size, images_source):
	batch_images = []
	batch_labels = []
	for i in range(batch_size):
		class_index = random.randint(0,1)
		batch_images.append(read_images(os.path.join(images_source, str(class_index))))
		batch_labels.append(onehot(class_index))
	return batch_images, batch_labels
		

def read_images(images_folder):
	image_path = os.path.join(images_folder,random.choice(os.listdir(images_folder)))
	im_array = preprocess_image(image_path)
	return im_array


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

	#for i in range(3):
	#	mean = np.mean(img_c1_np[:,:,i])
	#	stddev = np.std(img_c1_np[:,:,i])
	#	img_c1_np[:,:,i] -= mean
	#	img_c1_np[:,:,i] /= stddev

	return cropped_im_array

if __name__ == "__main__":
	data_folder_task1 = './imagenet/task1'
	data_folder_task2 = './imagenet/task2'
	data_task1_len = len(os.listdir('./imagenet/task1/0'))+len(os.listdir('./imagenet/task1/1'))
	data_task2_len = len(os.listdir('./imagenet/task2/0'))+len(os.listdir('./imagenet/task2/1'))

	#Number of image per task
	img1, label1 = read_batch(data_task1_len, data_folder_task1)
	img2, label2 = read_batch(data_task2_len, data_folder_task2)
	img1.reshape((data_task1_len,224*224*3))
	print (np.shape(img1))


