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

	# random 244x224 patch
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

def main(_):
	data_folder_task1 = './imagenet/task1'
	data_folder_task2 = './imagenet/task2'
	data_task1_len = len(os.listdir('./imagenet/task1/0'))+len(os.listdir('./imagenet/task1/1'))
	data_task2_len = len(os.listdir('./imagenet/task2/0'))+len(os.listdir('./imagenet/task2/1'))

	#Number of image per task
	img1, label1 = read_batch(data_task1_len, data_folder_task1)
	img2, label2 = read_batch(data_task2_len, data_folder_task2)
	# img1.reshape((data_task1_len,224*224*3))
	# print (np.shape(img1))

	## TASK 1
	sess = tf.InteractiveSession()
	# Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 224*224*3])
		y_ = tf.placeholder(tf.float32, [None, 2])

	geopath = pathnet.geopath_initializer(FLAGS.L,FLAGS.M);

	print 
	# fixed weights list
	fixed_list=np.ones((FLAGS.L,FLAGS.M),dtype=str);
	for i in range(FLAGS.L):
		for j in range(FLAGS.M):
			fixed_list[i,j]='0';    
	# Hidden Layers
	weights_list=np.zeros((FLAGS.L,FLAGS.M),dtype=object);
	biases_list=np.zeros((FLAGS.L,FLAGS.M),dtype=object);
	for i in range(FLAGS.L):
		for j in range(FLAGS.M):
			if(i==0):
				weights_list[i,j]=pathnet.module_weight_variable([224*224*3,FLAGS.filt]);
				biases_list[i,j]=pathnet.module_bias_variable([FLAGS.filt]);
	    	else:
	    		weights_list[i,j]=pathnet.module_weight_variable([FLAGS.filt,FLAGS.filt]);
	        	biases_list[i,j]=pathnet.module_bias_variable([FLAGS.filt]);

	for i in range(FLAGS.L):
		layer_modules_list=np.zeros(FLAGS.M,dtype=object);
		for j in range(FLAGS.M):
			if(i==0):
				layer_modules_list[j]=pathnet.module(x, weights_list[i,j], biases_list[i,j], 'layer'+str(i+1)+"_"+str(j+1))*geopath[i,j];
			else:
				layer_modules_list[j]=pathnet.module2(j,net, weights_list[i,j], biases_list[i,j], 'layer'+str(i+1)+"_"+str(j+1))*geopath[i,j];
		net=np.sum(layer_modules_list)/FLAGS.M;
	#net=net/FLAGS.M;  
	  # Output Layer
	output_weights=pathnet.module_weight_variable([FLAGS.filt,2]);
	output_biases=pathnet.module_bias_variable([2]);
	y = pathnet.nn_layer(net,output_weights,output_biases,'output_layer');

	# Cross Entropy
	with tf.name_scope('cross_entropy'):
		diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
	with tf.name_scope('total'):
	    cross_entropy = tf.reduce_mean(diff)
	tf.summary.scalar('cross_entropy', cross_entropy)
	  
	# Need to learn variables
	var_list_to_learn=[]+output_weights+output_biases;
	for i in range(FLAGS.L):
		for j in range(FLAGS.M):
			if (fixed_list[i,j]=='0'):
				var_list_to_learn+=weights_list[i,j]+biases_list[i,j];

	# GradientDescent 
	with tf.name_scope('train'):
		train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy,var_list=var_list_to_learn);

	# Accuracy 
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)



	# Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train1', sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test1')
	tf.global_variables_initializer().run()

	# Generating randomly geopath
	geopath_set=np.zeros(FLAGS.candi,dtype=object);
	for i in range(FLAGS.candi):
		geopath_set[i]=pathnet.get_geopath(FLAGS.L,FLAGS.M,FLAGS.N);
	  
	# parameters placeholders and ops 
	var_update_ops=np.zeros(len(var_list_to_learn),dtype=object);
	var_update_placeholders=np.zeros(len(var_list_to_learn),dtype=object);
	for i in range(len(var_list_to_learn)):
	    var_update_placeholders[i]=tf.placeholder(var_list_to_learn[i].dtype,shape=var_list_to_learn[i].get_shape());
	    var_update_ops[i]=var_list_to_learn[i].assign(var_update_placeholders[i]);
	 
	# geopathes placeholders and ops 
	geopath_update_ops=np.zeros((len(geopath),len(geopath[0])),dtype=object);
	geopath_update_placeholders=np.zeros((len(geopath),len(geopath[0])),dtype=object);
	for i in range(len(geopath)):
	    for j in range(len(geopath[0])):
	      	geopath_update_placeholders[i,j]=tf.placeholder(geopath[i,j].dtype,shape=geopath[i,j].get_shape());
	      	geopath_update_ops[i,j]=geopath[i,j].assign(geopath_update_placeholders[i,j]);
	     
	acc_geo=np.zeros(FLAGS.B,dtype=float); 
	summary_geo=np.zeros(FLAGS.B,dtype=object); 
	for i in range(FLAGS.max_steps):
	# Select Candidates to Tournament
	    compet_idx=range(FLAGS.candi);
	    np.random.shuffle(compet_idx);
	    compet_idx=compet_idx[:FLAGS.B];
	    # Learning & Evaluating
	    for j in range(len(compet_idx)):
	      	# Shuffle the data
	      	idx=range(len(tr_data1));
	      	np.random.shuffle(idx);
	      	tr_data1=tr_data1[idx];tr_label1=tr_label1[idx];
	      	# Insert Candidate
	      	pathnet.geopath_insert(sess,geopath_update_placeholders,geopath_update_ops,geopath_set[compet_idx[j]],FLAGS.L,FLAGS.M);
	      	acc_geo_tr=0;
	      	for k in range(FLAGS.T):
	        	summary_geo_tr, _, acc_geo_tmp = sess.run([merged, train_step,accuracy], feed_dict={x:tr_data1[k*FLAGS.batch_num:(k+1)*FLAGS.batch_num,:],y_:tr_label1[k*FLAGS.batch_num:(k+1)*FLAGS.batch_num,:]});
	        	acc_geo_tr+=acc_geo_tmp;
	      	acc_geo[j]=acc_geo_tr/FLAGS.T;
	      	summary_geo[j]=summary_geo_tr;
	    # Tournament
	    winner_idx=np.argmax(acc_geo);
	    acc=acc_geo[winner_idx];
	    summary=summary_geo[winner_idx];
	    # Copy and Mutation
	    for j in range(len(compet_idx)):
	      	if(j!=winner_idx):
	        	geopath_set[compet_idx[j]]=np.copy(geopath_set[compet_idx[winner_idx]]);
	        	geopath_set[compet_idx[j]]=pathnet.mutation(geopath_set[compet_idx[j]],FLAGS.L,FLAGS.M,FLAGS.N);
	    train_writer.add_summary(summary, i);
	    print('Training Accuracy at step %s: %s' % (i, acc));
	    if(acc >= 0.99):
	      	print('Learning Done!!');
	      	print('Optimal Path is as followed.');
	      	print(geopath_set[compet_idx[winner_idx]]);
	      	task1_optimal_path=geopath_set[compet_idx[winner_idx]];
	     	break;
	    """
	    geopath_sum=np.zeros((len(geopath),len(geopath[0])),dtype=float);
	    for j in range(len(geopath_set)):
	      for k in range(len(geopath)):
	        for l in range(len(geopath[0])):
	          geopath_sum[k][l]+=geopath_set[j][k][l];
	    print(geopath_sum);
	    """    
	iter_task1=i;    
	  
	# Fix task1 Optimal Path
	for i in range(FLAGS.L):
	    for j in range(FLAGS.M):
	      	if(task1_optimal_path[i,j]==1.0):
	        	fixed_list[i,j]='1';
	  
	# Get variables of fixed list
	var_list_to_fix=[];
	#var_list_to_fix=[]+output_weights+output_biases;
	for i in range(FLAGS.L):
	    for j in range(FLAGS.M):
	      	if(fixed_list[i,j]=='1'):
	        	var_list_to_fix+=weights_list[i,j]+biases_list[i,j];
	var_list_fix=pathnet.parameters_backup(var_list_to_fix);


	# parameters placeholders and ops 
	var_fix_ops=np.zeros(len(var_list_to_fix),dtype=object);
	var_fix_placeholders=np.zeros(len(var_list_to_fix),dtype=object);
	for i in range(len(var_list_to_fix)):
	    var_fix_placeholders[i]=tf.placeholder(var_list_to_fix[i].dtype,shape=var_list_to_fix[i].get_shape());
	    var_fix_ops[i]=var_list_to_fix[i].assign(var_fix_placeholders[i]);







if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--learning_rate', type=float, default=0.2,
                      help='Initial learning rate')
  parser.add_argument('--max_steps', type=int, default=500,
                      help='Number of steps to run trainer.')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--svhn_data_dir', type=str, default='/tmp/tensorflow/svhn/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--cifar_data_dir', type=str, default='/tmp/cifar10_data/cifar-10-batches-bin/',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/pathnet/',
                      help='Summaries log directry')
  parser.add_argument('--M', type=int, default=20,
                      help='The Number of Modules per Layer')
  parser.add_argument('--L', type=int, default=3,
                      help='The Number of Layers')
  parser.add_argument('--N', type=int, default=5,
                      help='The Number of Selected Modules per Layer')
  parser.add_argument('--T', type=int, default=50,
                      help='The Number of epoch per each geopath')
  parser.add_argument('--batch_num', type=int, default=16,
                      help='The Number of batches per each geopath')
  parser.add_argument('--filt', type=int, default=20,
                      help='The Number of Filters per Module')
  parser.add_argument('--candi', type=int, default=20,
                      help='The Number of Candidates of geopath')
  parser.add_argument('--B', type=int, default=2,
                      help='The Number of Candidates for each competition')
  parser.add_argument('--cifar_first', type=int, default=1,
                      help='If that is True, then cifar10 is first task.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
