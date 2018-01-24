from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np

def parameters_backup(var_list_to_learn):
  var_list_backup=np.zeros(len(var_list_to_learn),dtype=object);
  for i in range(len(var_list_to_learn)):
    var_list_backup[i]=var_list_to_learn[i].eval();
  return var_list_backup;

def parameters_update(sess,var_update_placeholders,var_update_ops,var_list_backup):
  for i in range(len(var_update_placeholders)):
    sess.run(var_update_ops[i],{var_update_placeholders[i]:var_list_backup[i]});
    
def geopath_insert(sess,geopath_update_placeholders,geopath_update_ops,candi,L,M):
  for i in range(L):
    for j in range(M):
      sess.run(geopath_update_ops[i,j],{geopath_update_placeholders[i,j]:candi[i,j]});

def geopath_initializer(L,M):
  geopath=np.zeros((L,M),dtype=object);
  for i in range(L):
    for j in range(M):
      geopath[i,j]=tf.Variable(1.0);
  return geopath;

def mutation(geopath,L,M,N):
  for i in range(L):
    for j in range(M):
      if(geopath[i,j]==1):
        rand_value=int(np.random.rand()*L*N);
        if(rand_value<=1):
          geopath[i,j]=0;
          rand_value2=np.random.randint(-2,2);
          rand_value2=rand_value2-2;
          if(((j+rand_value2)>=0)&((j+rand_value2)<M)):
            geopath[i,j+rand_value2]=1;
          else:
            if((j+rand_value2)<0):
              geopath[i,0]=1;
            elif((j+rand_value2)>=M):
              geopath[i,M-1]=1;
  return geopath;

def select_two_candi(M):
  selected=np.zeros(2,dtype=int);
  j=0;
  while j<=2:
    rand_value=int(np.random.rand()*M);
    if(j==0):
      selected[j]=rand_value;j+=1;
    else:
      if(selected[0]!=rand_value):
        selected[j]=rand_value;j+=1;
        break;
  return selected[0],selected[1];
  
def get_geopath(L,M,N):
  geopath=np.zeros((L,M),dtype=float);
  for i in range(L):
    j=0;
    #Active module # can be smaller than N
    while j<N:
      rand_value=int(np.random.rand()*M);
      if(geopath[i,rand_value]==0.0):
        geopath[i,rand_value]=1.0;j+=1;
  return geopath;
      

def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def module_weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return [tf.Variable(initial)];

def module_bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return [tf.Variable(initial)];

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

# input layer module, in another word, first hidden layer
def module(input_tensor, filt_num, is_active, layer_name, act=tf.nn.relu):
  # init
  weights=module_weight_variable([int(input_tensor.shape[-1]), filt_num])
  biases=module_bias_variable([filt_num])

  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      variable_summaries(weights[0])
    with tf.name_scope('biases'):
      variable_summaries(biases[0])
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights[0]) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations * is_active, weights, biases

# hidden layer module, include three kinds of modules in a layer
def module2(i,input_tensor, filt_num, is_active, layer_name, act=tf.nn.relu):
  # init
  weights = module_weight_variable([filt_num, filt_num])
  biases = module_bias_variable([filt_num])
  
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # Skip Layer
    if(i%3==0):
      return input_tensor * is_active, weights, biases
    # Linear Layer with relu
    elif(i%3==1):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        variable_summaries(weights[0])
      with tf.name_scope('biases'):
        variable_summaries(biases[0])
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights[0]) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations * is_active, weights, biases
    # Residual Layer with relu
    elif(i%3==2):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        variable_summaries(weights[0])
      with tf.name_scope('biases'):
        variable_summaries(biases[0])
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights[0]) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')+input_tensor
      tf.summary.histogram('activations', activations)
      return activations * is_active, weights, biases

def conv_module(input_tensor, weights, biases, stride, layer_name, act=tf.nn.relu):
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      variable_summaries(weights[0])
    with tf.name_scope('biases'):
      variable_summaries(biases[0])
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.nn.conv2d(input_tensor,weights[0],strides=[1,stride,stride,1],padding="VALID") + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations
 
def nn_layer(input_tensor, weights, biases, layer_name, act=tf.nn.relu):
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      variable_summaries(weights[0])
    with tf.name_scope('biases'):
      variable_summaries(biases[0])
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights[0]) + biases
      tf.summary.histogram('pre_activations', preactivate)
    return preactivate;


def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def _conv_layer(layer_name, input_tensor, filters, size, stride, padding,freeze=False, xavier=False, relu=True, stddev=0.001):
  with tf.variable_scope(layer_name) as scope:
    channels = input_tensor.get_shape()[3]
    # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
    # shape [h, w, in, out]
    kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
    bias_init = tf.constant_initializer(0.0)
    kernel = _variable_with_weight_decay('kernels', shape=[size, size, int(channels), filters],
      wd=0.0001, initializer=kernel_init, trainable=(not freeze))
    biases = _variable_on_device('biases', [filters], bias_init, 
                                trainable=(not freeze))
    conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding=padding,name='convolution')
    conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
    if relu:
      out = tf.nn.relu(conv_bias, 'relu')
    else:
      out = conv_bias
    return out

def _pooling_layer( layer_name, input_tensor, size, stride, padding='VALID'):
  with tf.variable_scope(layer_name) as scope:
    out =  tf.nn.max_pool(input_tensor, ksize=[1, size, size, 1], strides=[1, stride, stride, 1],padding=padding)
    activation_size = np.prod(out.get_shape().as_list()[1:])
    return out


#resnet for convolution
def res_conv_module(layer_name, input_tensor,  stddev=0.01,freeze=False):
  filter_num = tf.shape(input_tensor)[3]
  result_of_first_layer = _conv_layer(layer_name+'/firstlayer', input_tensor, filters = filter_num, size = 3, stride = 1,
    padding = 'SAME', stddev = stddev, freeze = freeze)
  result_of_second_layer = _conv_layer(layer_name+'/secondlayer', result_of_first_layer, filters = filter_num, size = 3, stride = 1,
    padding = 'SAME', stddev = stddev, freeze = freeze)
  return input_tensor + result_of_second_layer
  

#fire module
def fire_layer(layer_name, input_tensor, s1x1, e1x1, e3x3, stddev=0.01,freeze=False):
  sq1x1 = _conv_layer(layer_name+'/squeeze1x1', input_tensor, filters=s1x1, size=1, stride=1,
    padding='SAME', stddev=stddev, freeze=freeze)
  ex1x1 = _conv_layer(layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
    padding='SAME', stddev=stddev, freeze=freeze)
  ex3x3 = _conv_layer(layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
    padding='SAME', stddev=stddev, freeze=freeze)
  return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

#REF module
def REF_module():

