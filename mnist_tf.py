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

def create_network(learning_rate=1e-3):
  class Net:
    # layer 0
    x = tf.placeholder(tf.float32, shape=[None, 392])
    y_ = tf.placeholder(tf.float32, shape=[None, 14*28])
    x_image = tf.reshape(x, [-1,28,14,1])
    y_image = tf.reshape(y_, [-1,28,14,1])
    #input_summary = tf.image_summary('input image', x_image)
    #target_summary = tf.image_summary('target image', y_image)

    # layer 1
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    w1_hist = tf.histogram_summary('W_conv1 weights', W_conv1)
    b1_hist = tf.histogram_summary('b_conv1 biases', b_conv1)
    h1_hist = tf.histogram_summary('h_conv1 activations', h_conv1)
    '''

    # layer 2
    W_conv2 = weight_variable([3, 3, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    #w2_hist = tf.histogram_summary('W_conv2 weights', W_conv2)
    #b2_hist = tf.histogram_summary('b_conv2 biases', b_conv2)
    #h2_hist = tf.histogram_summary('h_conv2 activations', h_conv2)
    # layer 3
    W_conv3 = weight_variable([3, 3, 32, 32])
    b_conv3 = bias_variable([32])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

    w3_hist = tf.histogram_summary('W_conv3 weights', W_conv3)
    b3_hist = tf.histogram_summary('b_conv3 biases', b_conv3)
    h3_hist = tf.histogram_summary('h_conv3 activations', h_conv3)


    # layer 4
    W_conv4 = weight_variable([3, 3, 32, 32])
    b_conv4 = bias_variable([32])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    w4_hist = tf.histogram_summary('W_conv4 weights', W_conv4)
    b4_hist = tf.histogram_summary('b_conv4 biases', b_conv4)
    h4_hist = tf.histogram_summary('h_conv4 activations', h_conv4)


    # layer 5
    W_conv5 = weight_variable([3, 3, 32, 32])
    b_conv5 = bias_variable([32])
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

    w5_hist = tf.histogram_summary('W_conv5 weights', W_conv5)
    b5_hist = tf.histogram_summary('b_conv5 biases', b_conv5)
    h5_hist = tf.histogram_summary('h_conv5 activations', h_conv5)

    # layer 6
    W_conv6 = weight_variable([3, 3, 32, 32])
    b_conv6 = bias_variable([32])
    h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

    w6_hist = tf.histogram_summary('W_conv6 weights', W_conv6)
    b6_hist = tf.histogram_summary('b_conv6 biases', b_conv6)
    h6_hist = tf.histogram_summary('h_conv6 activations', h_conv6)

    # layer 7
    W_conv7 = weight_variable([3, 3, 32, 32])
    b_conv7 = bias_variable([32])
    h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv7) + b_conv7)

    w7_hist = tf.histogram_summary('W_conv7 weights', W_conv7)
    b7_hist = tf.histogram_summary('W_conv7 biases', W_conv7)
    h7_hist = tf.histogram_summary('h_conv7 activations', h_conv7)


    # layer 8
    W_conv8 = weight_variable([3, 3, 32, 32])
    b_conv8 = bias_variable([32])
    h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)

    w8_hist = tf.histogram_summary('W_conv8 weights', W_conv8)
    b8_hist = tf.histogram_summary('b_conv8 biases', b_conv8)
    h8_hist = tf.histogram_summary('h_conv8 activations', h_conv8)
    '''

    # reshape vector from convolution
    dim1 = 28 #- 5*2
    dim2 = 14 #- 5*2
    dimTotal = dim1 * dim2 * 32
    h_conv_flat = tf.reshape(h_conv1, [-1, dimTotal])
    
    # densely connected layer
    W_fc1 = weight_variable([dimTotal, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv_flat, W_fc1) + b_fc1)

    #w_fc1_hist = tf.histogram_summary('W_fc1 weights', W_fc1)
    #b_fc1_hist = tf.histogram_summary('b_fc1 biases', b_fc1)
    #h_fc1_hist = tf.histogram_summary('h_fc1 activations', h_fc1)

    # drop out layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # read out layer
    W_fc2 = weight_variable([1024, 14 * 28])
    b_fc2 = bias_variable([14 * 28])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    #w_fc2_hist = tf.histogram_summary('W_fc2 weights', W_fc2)
    #b_fc2_hist = tf.histogram_summary('b_fc2 biases', b_fc2)
    #y_conv_hist = tf.histogram_summary('y_conv activations', y_conv)

    cost = tf.reduce_mean(tf.mul(y_conv - y_, y_conv - y_)) #+ 0.0001*tf.reduce_sum(tf.abs(W_fc2))
    mse = tf.reduce_mean(tf.mul(y_conv - y_, y_conv - y_))
    #mse_summary = tf.scalar_summary('mse', mse)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    '''
    # tensorboard summaries
    summary_op = tf.merge_summary([w1_hist, b1_hist, h1_hist,
                                   w2_hist, b2_hist, h2_hist,
                                #   w3_hist, b3_hist, h3_hist,
                                #   w4_hist, b4_hist, h4_hist,
                                #   w5_hist, b5_hist, h5_hist,
                                #   w6_hist, b6_hist, h6_hist,
                                   w_fc1_hist, b_fc1_hist, h_fc1_hist,
                                   w_fc2_hist, b_fc2_hist, y_conv_hist,
                                   mse_summary
                                   ])

    image_summary_op = tf.merge_summary([input_summary,
                                         target_summary
                                         ])
    '''

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

  return Net()