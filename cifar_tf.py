# -----------------------------------------------------------------------------
#     Common Tensorflow code
# -----------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import os

# helper functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def create_autoencoder(maskVecXoneYzero, bottleneck, learning_rate=1e-3):
  class Net:
    epsilon = 1e-3      # define small epsilon for batch normalization
    keep_prob = tf.placeholder(tf.float32)      # keep prob for dropout

    # layer 0
    x = tf.placeholder(tf.float32, shape=[None, 3072])
    y = tf.placeholder(tf.float32, shape=[None, 3072])
    labels = tf.placeholder(tf.float32, shape=[None])
    labels = tf.cast(labels, tf.int64)
    x_image_pre = tf.reshape(x, [-1, 3, 32, 32])
    x_image = tf.transpose(x_image_pre, perm=[0, 2, 3, 1])

    # ---- encoder ----
    # layer 1 (conv - bn - relu - pool)
    W_e_conv1 = weight_variable([4, 4, 3, 64])
    b_e_conv1 = bias_variable([64])
    h_e_conv1_pre = conv2d(x_image, W_e_conv1)
    
    # batch_mean1, batch_var1 = tf.nn.moments(h_e_conv1_pre, axes=[0, 1, 2])
    # scale1 = tf.Variable(tf.ones([64]))
    # beta1 = tf.Variable(tf.zeros([64]))
    # bn1 = tf.nn.batch_normalization(h_e_conv1_pre, batch_mean1, batch_var1, beta1, scale1, epsilon)

    h_e_conv1 = tf.nn.relu(h_e_conv1_pre + b_e_conv1)

    h_pool1 = max_pool_2x2(h_e_conv1)

    # layer 2 (conv - bn - relu - pool)
    W_e_conv2 = weight_variable([4, 4, 64, 128])
    b_e_conv2 = bias_variable([128])
    h_e_conv2_pre = conv2d(h_pool1, W_e_conv2)

    # batch_mean2, batch_var2 = tf.nn.moments(h_e_conv2_pre, axes=[0, 1, 2])
    # scale2 = tf.Variable(tf.ones([128]))
    # beta2 = tf.Variable(tf.zeros([128]))
    # bn2 = tf.nn.batch_normalization(h_e_conv2_pre, batch_mean2, batch_var2, beta2, scale2, epsilon)

    h_e_conv2 = tf.nn.relu(h_e_conv2_pre + b_e_conv2)

    h_pool2 = max_pool_2x2(h_e_conv2)

    # layer 3 (conv - bn - relu - pool)
    W_e_conv3 = weight_variable([4, 4, 128, 256])
    b_e_conv3 = bias_variable([256])
    h_e_conv3_pre = conv2d(h_pool2, W_e_conv3)

    # batch_mean3, batch_var3 = tf.nn.moments(h_e_conv3_pre, axes=[0, 1, 2])
    # scale3 = tf.Variable(tf.ones([256]))
    # beta3 = tf.Variable(tf.zeros([256]))
    # bn3 = tf.nn.batch_normalization(h_e_conv3_pre, batch_mean3, batch_var3, beta3, scale3, epsilon)

    h_e_conv3 = tf.nn.relu(h_e_conv3_pre + b_e_conv3)

    h_pool3 = max_pool_2x2(h_e_conv3)

    # ---- fully connected layers ----
    dim_fc_in =  4 * 4 * 256
    W_e_fc1 = weight_variable([dim_fc_in, 1024])
    b_e_fc1 = bias_variable([1024])

    h_e_convs_flat = tf.reshape(h_pool3, [-1, dim_fc_in])
    h_e_fc1 = tf.nn.relu(tf.matmul(h_e_convs_flat, W_e_fc1) + b_e_fc1)

    # h_e_fc1_drop = tf.nn.dropout(h_e_fc1, keep_prob)

    W_e_fc2 = weight_variable([1024, bottleneck])
    b_e_fc2 = bias_variable([bottleneck])
    h_e_fc2 = tf.nn.relu(tf.matmul(h_e_fc1, W_e_fc2) + b_e_fc2)

    # h_e_fc2_drop = tf.nn.dropout(h_e_fc2, keep_prob)


    W = weight_variable([bottleneck, 10])
    b = bias_variable([10])
    h = (tf.matmul(h_e_fc2, W) + b)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=h)
    cost = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(labels, tf.argmax(h,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    '''
    # ---- decoder ----
    # ---- fully connected layers ----
    W_d_fc2 = weight_variable([bottleneck, 1024])
    b_d_fc2 = bias_variable([1024])
    h_d_fc2 = tf.nn.relu(tf.matmul(h_e_fc2_drop, W_d_fc2) + b_d_fc2)

    dim_fc_out =  4 * 4 * 256
    W_d_fc1 = weight_variable([1024, dim_fc_out])
    b_d_fc1 = bias_variable([dim_fc_out])
    h_d_fc1 = tf.nn.relu(tf.matmul(h_d_fc2, W_d_fc1) + b_d_fc1)
    h_d_fc1_drop = tf.nn.dropout(h_d_fc1, keep_prob)

    decoder_input = tf.reshape(h_d_fc1_drop, [-1, 4, 4, 256])

    # ---- convolutional layers ----
    strides=[1, 2, 2, 1]
    batch_size = tf.shape(x)[0]

    # layer 3
    W_d_conv3 = weight_variable([4, 4, 128, 256])
    b_d_conv3 = bias_variable([128])
    h_d_deconv3 = tf.nn.conv2d_transpose(decoder_input, W_d_conv3, output_shape=[batch_size, 8, 8, 128], strides=strides, padding='SAME')
    h_d_conv3 = tf.nn.relu(h_d_deconv3 + b_d_conv3)

    # layer 2
    W_d_conv2 = weight_variable([4, 4, 64, 128])
    b_d_conv2 = bias_variable([64])
    h_d_deconv2 = tf.nn.conv2d_transpose(h_d_conv3, W_d_conv2, output_shape=[batch_size, 16, 16, 64], strides=strides, padding='SAME')
    h_d_conv2 = tf.nn.relu(h_d_deconv2 + b_d_conv2)

    # layer 1
    W_d_conv1 = weight_variable([4, 4, 3, 64])
    b_d_conv1 = bias_variable([3])
    h_d_deconv1 = tf.nn.conv2d_transpose(h_d_conv2, W_d_conv1, output_shape=[batch_size, 32, 32, 3], strides=strides, padding='SAME')
    h_d_conv1 = tf.nn.relu(h_d_deconv1 + b_d_conv1)   

    h_d_conv1_rearrange = tf.transpose(h_d_conv1, perm=[0, 3, 1, 2])

    y_conv = tf.reshape(h_d_conv1_rearrange, [-1, 3072])

    cost = tf.reduce_mean(tf.mul(y_conv - y, y_conv - y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    '''

    '''
    # this is now [3072, batch_size]
    y_conv_transpose = tf.matrix_transpose(y_conv)
    y_transpose = tf.matrix_transpose(y)

    maskVec = np.reshape(maskVecXoneYzero, 3072)
    maskVecBool = np.ones((3072)) == (np.ones((3072)) - maskVec)
    y_conv_masked = tf.boolean_mask(y_conv_transpose, maskVecBool) 
    y_masked = tf.boolean_mask(y_transpose, maskVecBool)

    cost = tf.reduce_mean(tf.mul(y_conv_masked - y_masked, y_conv_masked - y_masked))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    '''
    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

  return Net()
