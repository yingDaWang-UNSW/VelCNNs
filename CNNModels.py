
"""functions used to construct different architectures  
velocity prediction in sparse porous media is best done without biases.
"""


import tensorflow as tf
import numpy as np
import pdb
import tensorflow.contrib.slim as slim
from tensorflow.keras.layers import UpSampling3D,UpSampling2D
def int_shape(x):
  return list(map(int, x.get_shape()))

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat(values=[x, -x], axis=axis))

def set_nonlinearity(name):
  if name == 'concat_elu':
    return concat_elu
  elif name == 'elu':
    return tf.nn.elu
  elif name == 'concat_relu':
    return tf.nn.crelu
  elif name == 'relu':
    return tf.nn.relu
  else:
    raise('nonlinearity ' + name + ' is not supported')

#def _activation_summary(x):
#  """Helper to create summaries for activations.
#  Creates a summary that provides a histogram of activations.
#  Creates a summary that measure the sparsity of activations.
#  Args:
#    x: Tensor
#  Returns:
#    nothing
#  """
#  tensor_name = x.op.name
#  tf.summary.histogram(tensor_name + '/activations', x)
#  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable(name, shape, initializer):
  """Helper to create a Variable.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  # getting rid of stddev for xavier ## testing this for faster convergence
  var = tf.get_variable(name, shape, initializer=initializer)
  return var

def conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None, nDims=2):
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = int(inputs.get_shape()[nDims+1])

    #biases = _variable('biases',[num_features],initializer=tf.contrib.layers.xavier_initializer())
    if nDims == 2:
        weights = _variable('weights', shape=[kernel_size,kernel_size,input_channels,num_features],initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
    elif nDims == 3:
        weights = _variable('weights', shape=[kernel_size,kernel_size,kernel_size,input_channels,num_features],initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv3d(inputs, weights, strides=[1, stride, stride, stride, 1], padding='SAME')
    conv_biased = conv#tf.nn.bias_add(conv, biases)
    if nonlinearity is not None:
      conv_biased = nonlinearity(conv_biased)
    return conv_biased

def transpose_conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None,nDims=2):
  with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
    input_channels = int(inputs.get_shape()[nDims+1])
    
    #biases = _variable('biases',[num_features],initializer=tf.contrib.layers.xavier_initializer())
    batch_size = tf.shape(inputs)[0]
    if nDims == 2:
        weights = _variable('weights', shape=[kernel_size,kernel_size,num_features,input_channels],initializer=tf.contrib.layers.xavier_initializer())
        output_shape = tf.stack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features]) 
        conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
    elif nDims ==3:
        weights = _variable('weights', shape=[kernel_size,kernel_size,kernel_size,num_features,input_channels],initializer=tf.contrib.layers.xavier_initializer())
        output_shape = tf.stack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, tf.shape(inputs)[3]*stride, num_features]) 
        conv = tf.nn.conv3d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,stride,1], padding='SAME')
    conv_biased = conv#tf.nn.bias_add(conv, biases)
    if nonlinearity is not None:
      conv_biased = nonlinearity(conv_biased)

    #reshape
    shape = int_shape(inputs)
    if nDims == 2:
        conv_biased = tf.reshape(conv_biased, [shape[0], shape[1]*stride, shape[2]*stride, num_features])
    elif nDims == 3:
        conv_biased = tf.reshape(conv_biased, [shape[0], shape[1]*stride, shape[2]*stride,shape[3]*stride, num_features])
    return conv_biased
    
def upsampling_conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None,nDims=2): # this trains way slower than transpose conv
  with tf.variable_scope('{0}_upsample_conv'.format(idx)) as scope:
    if nDims == 2:
        x = UpSampling2D(name='UpSampling2D')(inputs)
    elif nDims == 3:
        x = UpSampling3D(name='UpSampling3D')(inputs)
    x_1 = conv_layer(x, kernel_size, 1, num_features, 'upconv', nDims=nDims)
    return nonlinearity(x_1)

def fc_layer(inputs, hiddens, idx, nonlinearity=None, flat = False):
  with tf.variable_scope('{0}_fc'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs
    
    weights = _variable('weights', shape=[dim,hiddens],initializer=tf.contrib.layers.xavier_initializer())
    #biases = _variable('biases', [hiddens], initializer=tf.contrib.layers.xavier_initializer())
    output_biased = tf.matmul(inputs_processed,weights,name=str(idx)+'_fc')#tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')
    if nonlinearity is not None:
      output_biased = nonlinearity(ouput_biased)
    return output_biased

def nin(x, num_units, idx): # take input, 
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]),s[-1]]) # flatten everything except batchdim
    x = fc_layer(x, num_units, idx)
    return tf.reshape(x, s[:-1]+[num_units])
    
def denselayer(inputs, output_size):
    output = tf.layers.dense(inputs, output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return output
#def _phase_shift(I, r):
#  bsize, a, b, c = I.get_shape().as_list()
#  bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
#  X = tf.reshape(I, (bsize, a, b, r, r))
#  X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
#  X = tf.split(axis=1, num_or_size_splits=a, value=X)  # a, [bsize, b, r, r]
#  X = tf.concat(axis=2, values=[tf.squeeze(x) for x in X])  # bsize, b, a*r, r
#  X = tf.split(axis=1, num_or_size_splits=b, value=X)  # b, [bsize, a*r, r]
#  X = tf.concat(axis=2, values=[tf.squeeze(x) for x in X])  # bsize, a*r, b*r
#  return tf.reshape(X, (bsize, a*r, b*r, 1))

#def PS(X, r, depth):
#  Xc = tf.split(axis=3, num_or_size_splits=depth, value=X)
#  X = tf.concat(axis=3, values=[_phase_shift(x, r) for x in Xc])
#  return X

def res_block(x, a=None, filter_size=16, kernel_size=3, nonlinearity=concat_elu, keep_p=1.0, stride=1, gated=False, name="resnet", nDims=2): #this block is fully connected.
  orig_x = x
  orig_x_int_shape = int_shape(x)
  if orig_x_int_shape[nDims+1] == 1: # if input layer
    x_1 = conv_layer(x, kernel_size, stride, filter_size, name + '_conv_1', nDims=nDims)
  else:
    x_1 = conv_layer(nonlinearity(x), kernel_size, stride, filter_size, name + '_conv_1', nDims=nDims)
  if a is not None: # if a skip is active, do a thing
    shape_a = int_shape(a) 
    shape_x_1 = int_shape(x_1)
    if nDims ==2:
        a = tf.pad(
          a, [[0, 0], [0, shape_x_1[1]-shape_a[1]], [0, shape_x_1[2]-shape_a[2]],
          [0, 0]])
    elif nDims ==3:
        a = tf.pad(
          a, [[0, 0], [0, shape_x_1[1]-shape_a[1]], [0, shape_x_1[2]-shape_a[2]],[0, shape_x_1[3]-shape_a[3]],
          [0, 0]])
    #x_1 += conv_layer(nonlinearity(a), 1, 1, filter_size, name + '_nin')
    if nDims ==2:
        x_1 += nin(nonlinearity(a), filter_size, name + '_nin') # plus or multiply
    elif nDims ==3:
        x_1 += nin(nonlinearity(a), filter_size, name + '_nin') # plus or multiply
  x_1 = nonlinearity(x_1)
  if keep_p < 1.0:
    x_1 = tf.nn.dropout(x_1, keep_prob=keep_p)
  if not gated:
    x_2 = conv_layer(x_1, kernel_size, 1, filter_size, name + '_conv_2', nDims=nDims)
  else:
    x_2 = conv_layer(x_1, kernel_size, 1, filter_size*2, name + '_conv_2', nDims=nDims)
    x_2_1, x_2_2 = tf.split(axis=nDims+1,num_or_size_splits=2,value=x_2)
    x_2 = x_2_1 * tf.nn.sigmoid(x_2_2) # try tanh

  if int(orig_x.get_shape()[nDims]) > int(x_2.get_shape()[nDims]):
#    assert(int(orig_x.get_shape()[2]) == 2*int(x_2.get_shape()[2]), "res net block only supports stirde 2")
    if nDims==2:
        orig_x = tf.nn.avg_pool(orig_x, [1,2,2,1], [1,2,2,1], padding='SAME')
    elif nDims==3:
        orig_x = tf.nn.avg_pool3d(orig_x, [1,2,2,2,1], [1,2,2,2,1], padding='SAME')

  # pad it
  out_filter = filter_size
  in_filter = int(orig_x.get_shape()[nDims+1])
  if out_filter > in_filter:
    if nDims==2:
        orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0],[(out_filter-in_filter), 0]])
    elif nDims==3:
        orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [0, 0],[(out_filter-in_filter), 0]])
  elif out_filter < in_filter:
    if nDims==2:
        x_2 = tf.pad(x_2, [[0, 0], [0, 0], [0, 0],[(in_filter-out_filter), 0]])
    elif nDims==3:
        x_2 = tf.pad(x_2, [[0, 0], [0, 0], [0, 0], [0, 0],[(in_filter-out_filter), 0]])
  return orig_x + x_2

def gatedResnetGenerator(inputs, nr_res_blocks=1, keep_prob=1.0, nonlinearity_name='concat_elu', gated=True, filter_size = 8, kernel_size = 3, nDims = 2, outputType = 'vel'): # this is just unet with special resblocks
  """Builds conv part of net.
  Args:
    inputs: input images
    keep_prob: dropout layer
  """
  nonlinearity = set_nonlinearity(nonlinearity_name)
  # store for as
  a = []
  # res_1
  x = inputs
  for i in range(nr_res_blocks):
    x = res_block(x, filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_1_" + str(i), nDims=nDims)
  # res_2
  a.append(x)
  filter_size = 2 * filter_size
  x = res_block(x, filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_2_downsample", nDims=nDims)
  for i in range(nr_res_blocks):
    x = res_block(x, filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_2_" + str(i), nDims=nDims)
  # res_3
  a.append(x)
  filter_size = 2 * filter_size
  x = res_block(x, filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_3_downsample", nDims=nDims)
  for i in range(nr_res_blocks):
    x = res_block(x, filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_3_" + str(i), nDims=nDims)
  # res_4
  a.append(x)
  filter_size = 2 * filter_size
  x = res_block(x, filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_4_downsample", nDims=nDims)
  for i in range(nr_res_blocks):
    x = res_block(x, filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_4_" + str(i), nDims=nDims)
  # res_4
  a.append(x)
  filter_size = 2 * filter_size
  x = res_block(x, filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated, name="resnet_5_downsample", nDims=nDims)
  for i in range(nr_res_blocks):
    x = res_block(x, filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_5_" + str(i), nDims=nDims)
  # res_up_1
  filter_size = int(filter_size /2)
  x = transpose_conv_layer(x, kernel_size, 2, filter_size, "up_conv_1", nonlinearity=None, nDims=nDims)
  #x = PS(x,2,512)
  for i in range(nr_res_blocks):
    if i == 0:
      x = res_block(x, a=a[-1], filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_1_" + str(i), nDims=nDims)
    else:
      x = res_block(x, filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_1_" + str(i), nDims=nDims)
  # res_up_1
  filter_size = int(filter_size /2)
  x = transpose_conv_layer(x, kernel_size, 2, filter_size, "up_conv_2", nonlinearity=None, nDims=nDims)
  #x = PS(x,2,512)
  for i in range(nr_res_blocks):
    if i == 0:
      x = res_block(x, a=a[-2], filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_2_" + str(i), nDims=nDims)
    else:
      x = res_block(x, filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_2_" + str(i), nDims=nDims)
  filter_size = int(filter_size /2)
  x = transpose_conv_layer(x, kernel_size, 2, filter_size, "up_conv_3", nonlinearity=None, nDims=nDims)
  #x = PS(x,2,512)
  for i in range(nr_res_blocks):
    if i == 0:
      x = res_block(x, a=a[-3], filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_3_" + str(i), nDims=nDims)
    else:
      x = res_block(x, filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_3_" + str(i), nDims=nDims)
 
  filter_size = int(filter_size /2)
  x = transpose_conv_layer(x, kernel_size, 2, filter_size, "up_conv_4", nonlinearity=None, nDims=nDims)
  #x = PS(x,2,512)
  for i in range(nr_res_blocks):
    if i == 0:
      x = res_block(x, a=a[-4], filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_4_" + str(i), nDims=nDims)
    else:
      x = res_block(x, filter_size=filter_size, kernel_size=kernel_size, nonlinearity=nonlinearity, keep_p=keep_prob, gated=gated, name="resnet_up_4_" + str(i), nDims=nDims)
#  if nDims == 3:
#    x=x*tf.tile(inputs,[1,1,1,1,filter_size])
#  elif nDims ==2:
#    x=x*tf.tile(inputs,[1,1,1,filter_size])
  if outputType == 'vel':
    x = conv_layer(x, kernel_size, 1, nDims, "last_conv", nDims=nDims)
  elif outputType == 'fq':
    x = conv_layer(x, kernel_size, 1, 19, "last_conv", nDims=nDims)
  elif outputType == 'velP':
    x = conv_layer(x, kernel_size, 1, nDims+1, "last_conv", nDims=nDims)
  elif outputType == 'P':
    x = conv_layer(x, kernel_size, 1, 1, "last_conv", nDims=nDims)
  elif outputType == 'k':
    x = slim.flatten(x)
    x = denselayer(x, 16)
    x = tf.maximum(x, 0.2 * x)
    x = denselayer(x, 1)
#  # this skip connection is a bit controversial
#  if nDims == 3:
#    x=x*tf.tile(inputs,[1,1,1,1,nDims])
#  elif nDims ==2:
#    x=x*tf.tile(inputs,[1,1,1,nDims])
  return x
  
  

def discriminatorTF(input_disc, kernel, filters, is_train=True, reuse=False, nDims=2):

    stride=1
    biasFlag=True
    kernelInit=None
    
    
    
    def discriminator_block(inputs, numFilters, kernelSize, stride, scope, reuse, nDims):
        with tf.variable_scope(scope):
            #net = conv3dydw(inputs, numInFilters, numFilters, kernelSize, stride, name='DConv1', reuse=reuse, trainable=is_train, padding='same', use_bias=biasFlag, kernel_initializer=kernelInit)
            net = conv_layer(inputs, kernelSize, stride, numFilters, scope + '_conv', nDims=nDims)
            net = batchnorm(net, is_train)
            net = lrelu2(net)
        return net
      
    def batchnorm(inputs, is_training):
        return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                            scale=False, fused=True, is_training=is_training)
    # Our dense layer
    def denselayer(inputs, output_size):
        output = tf.layers.dense(inputs, output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        return output
        
    def lrelu2(x):
        return tf.maximum(x, 0.2 * x)
        
    with tf.variable_scope("discriminator", reuse=reuse):
        x = input_disc
        # cripple the disriminator with noise
        #x = gaussian_noise_layer(x, 0)
        x = conv_layer(x, kernel, stride, filters, 'conv_1', nDims=nDims)
        #x =  tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel, strides=stride, padding='SAME')(x)
        x = lrelu2(x)
        # The discriminator block part
        # block 1
        x = discriminator_block(x, filters, kernel, 2, 'disblock_1', reuse, nDims)

        # block 2
        x = discriminator_block(x, filters*2, kernel, 1, 'disblock_2', reuse, nDims)

        # block 3
        x = discriminator_block(x, filters*2, kernel, 2, 'disblock_3', reuse, nDims)

        # block 4
        x = discriminator_block(x, filters*4, kernel, 1, 'disblock_4', reuse, nDims)

        # block 5
        x = discriminator_block(x, filters*4, kernel, 2, 'disblock_5', reuse, nDims)

        # block 6
        x = discriminator_block(x, filters*8, kernel, 1, 'disblock_6', reuse, nDims)

        # block_7
        x = discriminator_block(x, filters*8, kernel, 2, 'disblock_7', reuse, nDims)

        # The dense layer 1
        with tf.variable_scope('dense_layer_1'):
            x = slim.flatten(x)
            x = denselayer(x, filters*16)
            x = lrelu2(x)

        # The dense layer 2
        with tf.variable_scope('dense_layer_2'):
            x = denselayer(x, 1)
            logits = x
            x = tf.nn.sigmoid(x)
    return x, logits


