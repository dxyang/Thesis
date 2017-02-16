# -----------------------------------------------------------------------------
#     Common Tensorflow code
# -----------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import os
import mnist_preprocessing

# helper functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_nopad(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def create_network_basic(learning_rate=1e-3):
  class Net:
    epsilon = 1e-3      # define small epsilon for batch normalization
    keep_prob = tf.placeholder(tf.float32)      # keep prob for dropout

    # layer 0
    x = tf.placeholder(tf.float32, shape=[None, 392])
    y_ = tf.placeholder(tf.float32, shape=[None, 14*28])
    x_image = tf.reshape(x, [-1,14,28,1])
    y_image = tf.reshape(y_, [-1,14,28,1])
    input_summary = tf.image_summary('input image', x_image)
    target_summary = tf.image_summary('target image', y_image)
    
    # layer 1
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    w1_hist = tf.histogram_summary('W_conv1 weights', W_conv1)
    b1_hist = tf.histogram_summary('b_conv1 biases', b_conv1)
    h1_hist = tf.histogram_summary('h_conv1 activations', h_conv1)
    
    # layer 2
    W_conv2 = weight_variable([3, 3, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    w2_hist = tf.histogram_summary('W_conv2 weights', W_conv2)
    b2_hist = tf.histogram_summary('b_conv2 biases', b_conv2)
    h2_hist = tf.histogram_summary('h_conv2 activations', h_conv2)
    '''
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
    '''
    # reshape vector from convolution
    dim1 = 28 #- 5*2
    dim2 = 14 #- 5*2
    dimTotal = dim1 * dim2 * 32
    h_conv_flat = tf.reshape(h_conv2, [-1, dimTotal])
    
    # densely connected layer
    W_fc1 = weight_variable([dimTotal, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv_flat, W_fc1) + b_fc1)

    w_fc1_hist = tf.histogram_summary('W_fc1 weights', W_fc1)
    b_fc1_hist = tf.histogram_summary('b_fc1 biases', b_fc1)
    h_fc1_hist = tf.histogram_summary('h_fc1 activations', h_fc1)

    # drop out layer
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # read out layer
    W_fc2 = weight_variable([1024, 14 * 28])
    b_fc2 = bias_variable([14 * 28])
    y_conv = tf.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

    w_fc2_hist = tf.histogram_summary('W_fc2 weights', W_fc2)
    b_fc2_hist = tf.histogram_summary('b_fc2 biases', b_fc2)
    y_conv_hist = tf.histogram_summary('y_conv activations', y_conv)

    cost = tf.reduce_mean(tf.mul(y_conv - y_, y_conv - y_)) #+ 0.0001*tf.reduce_sum(tf.abs(W_fc2))
    cost_summary = tf.scalar_summary('cost', cost)
    mse = tf.reduce_mean(tf.mul(y_conv - y_, y_conv - y_))
    mse_summary = tf.scalar_summary('mse', mse)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # tensorboard summaries
    summary_op = tf.merge_summary([w1_hist, b1_hist, h1_hist,
                                   w2_hist, b2_hist, h2_hist,
                                #   w3_hist, b3_hist, h3_hist,
                                #   w4_hist, b4_hist, h4_hist,
                                #   w5_hist, b5_hist, h5_hist,
                                #   w6_hist, b6_hist, h6_hist,
                                   w_fc1_hist, b_fc1_hist, h_fc1_hist,
                                   w_fc2_hist, b_fc2_hist, y_conv_hist,
                                   mse_summary, cost_summary
                                   ])

    image_summary_op = tf.merge_summary([input_summary,
                                         target_summary
                                         ])

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

  return Net()

def create_network_batchnorm(learning_rate=1e-3):
  class Net:
    epsilon = 1e-3      # define small epsilon for batch normalization
    keep_prob = tf.placeholder(tf.float32)      # keep prob for dropout

    # layer 0
    x = tf.placeholder(tf.float32, shape=[None, 392])
    y_ = tf.placeholder(tf.float32, shape=[None, 14*28])
    x_image = tf.reshape(x, [-1,14,28,1])
    y_image = tf.reshape(y_, [-1,14,28,1])
    input_summary = tf.image_summary('input image', x_image)
    target_summary = tf.image_summary('target image', y_image)
    
    # ----- layer 1 -----
    # convolve
    W_conv1 = weight_variable([3, 3, 1, 32])
    h_conv1 = conv2d(x_image, W_conv1)
    
    # batch normalization
    batch_mean1, batch_var1 = tf.nn.moments(h_conv1, axes=[0, 1, 2])
    scale1 = tf.Variable(tf.ones([32]))
    beta1 = tf.Variable(tf.zeros([32]))
    bn1 = tf.nn.batch_normalization(h_conv1, batch_mean1, batch_var1, beta1, scale1, epsilon)

    # apply nonlinearity
    b_conv1 = bias_variable([32])
    h_lin1 = tf.nn.relu(bn1 + b_conv1)

    # apply dropout
    #h_lin1_drop = tf.nn.dropout(h_lin1, keep_prob)

    # summaries
    w1_hist = tf.histogram_summary('W_conv1 weights', W_conv1)
    b1_hist = tf.histogram_summary('b_conv1 biases', b_conv1)
    h1_hist = tf.histogram_summary('h_lin1 activations', h_lin1)
    scale1_hist = tf.histogram_summary('scale1 biases', scale1)
    beta1_hist = tf.histogram_summary('beta1 biases', beta1)
    
    # ----- layer 2 -----
    # convolve
    W_conv2 = weight_variable([3, 3, 32, 32])
    h_conv2 = conv2d(h_lin1, W_conv2)
    
    # batch normalization
    batch_mean2, batch_var2 = tf.nn.moments(h_conv2, axes=[0, 1, 2])
    scale2 = tf.Variable(tf.ones([32]))
    beta2 = tf.Variable(tf.zeros([32]))
    bn2 = tf.nn.batch_normalization(h_conv2, batch_mean2, batch_var2, beta2, scale2, epsilon)

    # apply nonlinearity
    b_conv2 = bias_variable([32])
    h_lin2 = tf.nn.relu(bn2 + b_conv2)

    # apply dropout
    #h_lin2_drop = tf.nn.dropout(h_lin2, keep_prob)

    # summaries
    w2_hist = tf.histogram_summary('W_conv2 weights', W_conv2)
    b2_hist = tf.histogram_summary('b_conv2 biases', b_conv2)
    h2_hist = tf.histogram_summary('h_conv2 activations', h_lin2)
    scale2_hist = tf.histogram_summary('scale2 biases', scale2)
    beta2_hist = tf.histogram_summary('beta2 biases', beta2)

    # ----- layer 3 -----
    # convolve
    W_conv3 = weight_variable([3, 3, 32, 32])
    h_conv3 = conv2d(h_lin2, W_conv3)
    
    # batch normalization
    batch_mean3, batch_var3 = tf.nn.moments(h_conv3, axes=[0, 1, 2])
    scale3 = tf.Variable(tf.ones([32]))
    beta3 = tf.Variable(tf.zeros([32]))
    bn3 = tf.nn.batch_normalization(h_conv3, batch_mean3, batch_var3, beta3, scale3, epsilon)

    # apply nonlinearity
    b_conv3 = bias_variable([32])
    h_lin3 = tf.nn.relu(bn3 + b_conv3)

    # apply dropout
    #h_lin3_drop = tf.nn.dropout(h_lin3, keep_prob)

    # summaries
    w3_hist = tf.histogram_summary('W_conv3 weights', W_conv3)
    b3_hist = tf.histogram_summary('b_conv3 biases', b_conv3)
    h3_hist = tf.histogram_summary('h_conv3 activations', h_lin3)
    scale3_hist = tf.histogram_summary('scale3 biases', scale3)
    beta3_hist = tf.histogram_summary('beta3 biases', beta3)

    # ----- layer 4 -----
    # convolve
    W_conv4 = weight_variable([3, 3, 32, 32])
    h_conv4 = conv2d(h_lin3, W_conv4)
    
    # batch normalization
    batch_mean4, batch_var4 = tf.nn.moments(h_conv4, axes=[0, 1, 2])
    scale4 = tf.Variable(tf.ones([32]))
    beta4 = tf.Variable(tf.zeros([32]))
    bn4 = tf.nn.batch_normalization(h_conv4, batch_mean4, batch_var4, beta4, scale4, epsilon)

    # apply nonlinearity
    b_conv4 = bias_variable([32])
    h_lin4 = tf.nn.relu(bn4 + b_conv4)

    # apply dropout
    #h_lin4_drop = tf.nn.dropout(h_lin4, keep_prob)

    # summaries
    w4_hist = tf.histogram_summary('W_conv4 weights', W_conv4)
    b4_hist = tf.histogram_summary('b_conv4 biases', b_conv4)
    h4_hist = tf.histogram_summary('h_conv4 activations', h_lin4)
    scale4_hist = tf.histogram_summary('scale4 biases', scale4)
    beta4_hist = tf.histogram_summary('beta4 biases', beta4)

    # ----- layer 5 -----
    # convolve
    W_conv5 = weight_variable([3, 3, 32, 32])
    h_conv5 = conv2d(h_lin4, W_conv5)
    
    # batch normalization
    batch_mean5, batch_var5 = tf.nn.moments(h_conv5, axes=[0, 1, 2])
    scale5 = tf.Variable(tf.ones([32]))
    beta5 = tf.Variable(tf.zeros([32]))
    bn5 = tf.nn.batch_normalization(h_conv5, batch_mean5, batch_var5, beta5, scale5, epsilon)

    # apply nonlinearity
    b_conv5 = bias_variable([32])
    h_lin5 = tf.nn.relu(bn5 + b_conv5)

    # apply dropout
    #h_lin5_drop = tf.nn.dropout(h_lin5, keep_prob)

    # summaries
    w5_hist = tf.histogram_summary('W_conv5 weights', W_conv5)
    b5_hist = tf.histogram_summary('b_conv5 biases', b_conv5)
    h5_hist = tf.histogram_summary('h_conv5 activations', h_conv5)
    scale5_hist = tf.histogram_summary('scale5 biases', scale5)
    beta5_hist = tf.histogram_summary('beta5 biases', beta5)

    # ----- layer 6 -----
    # convolve
    W_conv6 = weight_variable([3, 3, 32, 32])
    h_conv6 = conv2d(h_lin5, W_conv6)
    
    # batch normalization
    batch_mean6, batch_var6 = tf.nn.moments(h_conv6, axes=[0, 1, 2])
    scale6 = tf.Variable(tf.ones([32]))
    beta6 = tf.Variable(tf.zeros([32]))
    bn6 = tf.nn.batch_normalization(h_conv6, batch_mean6, batch_var6, beta6, scale6, epsilon)

    # apply nonlinearity
    b_conv6 = bias_variable([32])
    h_lin6 = tf.nn.relu(bn6 + b_conv6)

    # apply dropout
    #h_lin6_drop = tf.nn.dropout(h_lin6, keep_prob)

    # summaries
    w6_hist = tf.histogram_summary('W_conv6 weights', W_conv6)
    b6_hist = tf.histogram_summary('b_conv6 biases', b_conv6)
    h6_hist = tf.histogram_summary('h_conv6 activations', h_lin6)
    scale6_hist = tf.histogram_summary('scale6 biases', scale6)
    beta6_hist = tf.histogram_summary('beta6 biases', beta6)

    # ----------------------------------------------------------------------
    # reshape vector from convolution
    dim1 = 28 #- 5*2
    dim2 = 14 #- 5*2
    dimTotal = dim1 * dim2 * 32
    h_conv_flat = tf.reshape(h_lin6, [-1, dimTotal])
    
    # densely connected layer
    W_fc1 = weight_variable([dimTotal, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv_flat, W_fc1) + b_fc1)

    w_fc1_hist = tf.histogram_summary('W_fc1 weights', W_fc1)
    b_fc1_hist = tf.histogram_summary('b_fc1 biases', b_fc1)
    h_fc1_hist = tf.histogram_summary('h_fc1 activations', h_fc1)

    # drop out layer
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # read out layer
    W_fc2 = weight_variable([1024, 14 * 28])
    b_fc2 = bias_variable([14 * 28])
    y_conv = tf.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

    w_fc2_hist = tf.histogram_summary('W_fc2 weights', W_fc2)
    b_fc2_hist = tf.histogram_summary('b_fc2 biases', b_fc2)
    y_conv_hist = tf.histogram_summary('y_conv activations', y_conv)

    cost = tf.reduce_mean(tf.mul(y_conv - y_, y_conv - y_)) #+ 0.0001*tf.reduce_sum(tf.abs(W_fc2))
    cost_summary = tf.scalar_summary('cost', cost)
    mse = tf.reduce_mean(tf.mul(y_conv - y_, y_conv - y_))
    mse_summary = tf.scalar_summary('mse', mse)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # tensorboard summaries
    summary_op = tf.merge_summary([w1_hist, b1_hist, h1_hist, scale1_hist, beta1_hist,
                                   w2_hist, b2_hist, h2_hist, scale2_hist, beta2_hist,
                                   w3_hist, b3_hist, h3_hist, scale3_hist, beta3_hist,
                                   w4_hist, b4_hist, h4_hist, scale4_hist, beta4_hist,
                                   w5_hist, b5_hist, h5_hist, scale5_hist, beta5_hist,
                                   w6_hist, b6_hist, h6_hist, scale6_hist, beta6_hist,
                                   w_fc1_hist, b_fc1_hist, h_fc1_hist,
                                   w_fc2_hist, b_fc2_hist, y_conv_hist,
                                   mse_summary, cost_summary
                                   ])

    image_summary_op = tf.merge_summary([input_summary,
                                         target_summary
                                         ])

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

  return Net()

def create_network_fullyconnected(learning_rate=1e-3):
  class Net:
    keep_prob = tf.placeholder(tf.float32)      # keep prob for dropout

    # layer 0
    x = tf.placeholder(tf.float32, shape=[None, 392])
    y_ = tf.placeholder(tf.float32, shape=[None, 14*28])
    
    # ----- layer 1 -----
    W_1 = weight_variable([392, 392])
    b_1 = bias_variable([392])
    h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)
    h_1_drop = tf.nn.dropout(h_1, keep_prob)
    
    # ----- layer 2 -----
    W_2 = weight_variable([392, 392])
    b_2 = bias_variable([392])
    h_2 = tf.nn.relu(tf.matmul(h_1_drop, W_2) + b_2)
    h_2_drop = tf.nn.dropout(h_2, keep_prob)
    '''
    # ----- layer 3 -----
    W_3 = weight_variable([392, 392])
    b_3 = bias_variable([392])
    h_3 = tf.nn.relu(tf.matmul(h_2_drop, W_3) + b_3)
    h_3_drop = tf.nn.dropout(h_3, keep_prob)

    # ----- layer 4 -----
    W_4 = weight_variable([392, 392])
    b_4 = bias_variable([392])
    h_4 = tf.nn.relu(tf.matmul(h_3_drop, W_4) + b_4)
    h_4_drop = tf.nn.dropout(h_4, keep_prob)

    # ----- layer 5 -----
    W_5 = weight_variable([392, 392])
    b_5 = bias_variable([392])
    h_5 = tf.nn.relu(tf.matmul(h_4_drop, W_5) + b_5)
    h_5_drop = tf.nn.dropout(h_5, keep_prob)

    # ----- layer 6 -----
    W_6 = weight_variable([392, 392])
    b_6 = bias_variable([392])
    h_6 = tf.nn.relu(tf.matmul(h_5_drop, W_6) + b_6)
    h_6_drop = tf.nn.dropout(h_6, keep_prob)
    '''
    y_conv = h_2_drop

    cost = tf.reduce_mean(tf.mul(y_conv - y_, y_conv - y_)) #+ 0.0001*tf.reduce_sum(tf.abs(W_fc2))
    mse = tf.reduce_mean(tf.mul(y_conv - y_, y_conv - y_))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

  return Net()

def create_network_kinda_autoencoder(learning_rate=1e-3):
  class Net:
    epsilon = 1e-3      # define small epsilon for batch normalization
    keep_prob = tf.placeholder(tf.float32)      # keep prob for dropout

    # layer 0
    x = tf.placeholder(tf.float32, shape=[None, 392])
    y_ = tf.placeholder(tf.float32, shape=[None, 14*28])
    x_image = tf.reshape(x, [-1,14,28,1])
    #y_image = tf.reshape(y_, [-1,14,28,1])
    #input_summary = tf.image_summary('input image', x_image)
    #target_summary = tf.image_summary('target image', y_image)
    
    # ---- encoder ----
    # layer 1
    W_e_conv1 = weight_variable([3, 3, 1, 32])
    b_e_conv1 = bias_variable([32])
    h_e_conv1 = tf.nn.relu(conv2d(x_image, W_e_conv1) + b_e_conv1)

    #w1_e_hist = tf.histogram_summary('W_e_conv1 weights', W_e_conv1)
    #b1_e_hist = tf.histogram_summary('b_e_conv1 biases', b_e_conv1)
    #h1_e_hist = tf.histogram_summary('h_e_conv1 activations', h_e_conv1)
    
    # layer 2
    W_e_conv2 = weight_variable([3, 3, 32, 32])
    b_e_conv2 = bias_variable([32])
    h_e_conv2 = tf.nn.relu(conv2d(h_e_conv1, W_e_conv2) + b_e_conv2)

    #w2_e_hist = tf.histogram_summary('W_e_conv2 weights', W_e_conv2)
    #b2_e_hist = tf.histogram_summary('b_e_conv2 biases', b_e_conv2)
    #h2_e_hist = tf.histogram_summary('h_e_conv2 activations', h_e_conv2)

    # layer 3
    W_e_conv3 = weight_variable([3, 3, 32, 32])
    b_e_conv3 = bias_variable([32])
    h_e_conv3 = tf.nn.relu(conv2d(h_e_conv2, W_e_conv3) + b_e_conv3)

    # layer 4
    W_e_conv4 = weight_variable([3, 3, 32, 32])
    b_e_conv4 = bias_variable([32])
    h_e_conv4 = tf.nn.relu(conv2d(h_e_conv3, W_e_conv4) + b_e_conv4)

    # layer 5
    W_e_conv5 = weight_variable([3, 3, 32, 32])
    b_e_conv5 = bias_variable([32])
    h_e_conv5 = tf.nn.relu(conv2d(h_e_conv4, W_e_conv5) + b_e_conv5)

    # layer 6
    W_e_conv6 = weight_variable([3, 3, 32, 32])
    b_e_conv6 = bias_variable([32])
    h_e_conv6 = tf.nn.relu(conv2d(h_e_conv5, W_e_conv6) + b_e_conv6)

    # layer 7
    W_e_conv7 = weight_variable([3, 3, 32, 32])
    b_e_conv7 = bias_variable([32])
    h_e_conv7 = tf.nn.relu(conv2d(h_e_conv6, W_e_conv7) + b_e_conv7)

    # layer 8
    W_e_conv8 = weight_variable([3, 3, 32, 32])
    b_e_conv8 = bias_variable([32])
    h_e_conv8 = tf.nn.relu(conv2d(h_e_conv7, W_e_conv8) + b_e_conv8)

    # layer 9
    W_e_conv9 = weight_variable([3, 3, 32, 32])
    b_e_conv9 = bias_variable([32])
    h_e_conv9 = tf.nn.relu(conv2d(h_e_conv8, W_e_conv9) + b_e_conv9)

    # layer 8
    W_e_conv10 = weight_variable([3, 3, 32, 32])
    b_e_conv10 = bias_variable([32])
    h_e_conv10 = tf.nn.relu(conv2d(h_e_conv9, W_e_conv10) + b_e_conv10)

    # ---- decoder ----
    strides=[1, 1, 1, 1]
    batch_size = tf.shape(x)[0]

    # layer 1
    W_d_conv1 = weight_variable([3, 3, 32, 32])
    b_d_conv1 = bias_variable([32])
    h_d_deconv1 = tf.nn.conv2d_transpose(h_e_conv10, W_d_conv1, output_shape=[batch_size, 14, 28, 32], strides=strides, padding='SAME')
    h_d_conv1 = tf.nn.relu(h_d_deconv1 + b_d_conv1)

    #w1_d_hist = tf.histogram_summary('W_d_conv1 weights', W_d_conv1)
    #b1_d_hist = tf.histogram_summary('b_d_conv1 biases', b_d_conv1)
    #h1_d_hist = tf.histogram_summary('h_d_conv1 activations', h_d_conv1)
    
    # layer 2
    W_d_conv2 = weight_variable([3, 3, 32, 32])
    b_d_conv2 = bias_variable([1])
    h_d_deconv2 = tf.nn.conv2d_transpose(h_d_conv1, W_d_conv2, output_shape=[batch_size, 14, 28, 32], strides=strides, padding='SAME')
    h_d_conv2 = tf.nn.relu(h_d_deconv2 + b_d_conv2)

    #w2_d_hist = tf.histogram_summary('W_d_conv2 weights', W_d_conv2)
    #b2_d_hist = tf.histogram_summary('b_d_conv2 biases', b_d_conv2)
    #h2_d_hist = tf.histogram_summary('h_d_conv2 activations', h_d_conv2)

    # layer 3
    W_d_conv3 = weight_variable([3, 3, 32, 32])
    b_d_conv3 = bias_variable([1])
    h_d_deconv3 = tf.nn.conv2d_transpose(h_d_conv2, W_d_conv3, output_shape=[batch_size, 14, 28, 32], strides=strides, padding='SAME')
    h_d_conv3 = tf.nn.relu(h_d_deconv3 + b_d_conv3)

    # layer 4
    W_d_conv4 = weight_variable([3, 3, 32, 32])
    b_d_conv4 = bias_variable([1])
    h_d_deconv4 = tf.nn.conv2d_transpose(h_d_conv3, W_d_conv4, output_shape=[batch_size, 14, 28, 32], strides=strides, padding='SAME')
    h_d_conv4 = tf.nn.relu(h_d_deconv4 + b_d_conv4)

    # layer 5
    W_d_conv5 = weight_variable([3, 3, 32, 32])
    b_d_conv5 = bias_variable([1])
    h_d_deconv5 = tf.nn.conv2d_transpose(h_d_conv4, W_d_conv5, output_shape=[batch_size, 14, 28, 32], strides=strides, padding='SAME')
    h_d_conv5 = tf.nn.relu(h_d_deconv5 + b_d_conv5)

    # layer 6
    W_d_conv6 = weight_variable([3, 3, 32, 32])
    b_d_conv6 = bias_variable([1])
    h_d_deconv6 = tf.nn.conv2d_transpose(h_d_conv5, W_d_conv6, output_shape=[batch_size, 14, 28, 32], strides=strides, padding='SAME')
    h_d_conv6 = tf.nn.relu(h_d_deconv6 + b_d_conv6)

    # layer 7
    W_d_conv7 = weight_variable([3, 3, 32, 32])
    b_d_conv7 = bias_variable([1])
    h_d_deconv7 = tf.nn.conv2d_transpose(h_d_conv6, W_d_conv7, output_shape=[batch_size, 14, 28, 32], strides=strides, padding='SAME')
    h_d_conv7 = tf.nn.relu(h_d_deconv7 + b_d_conv7)

    # layer 8
    W_d_conv8 = weight_variable([3, 3, 32, 32])
    b_d_conv8 = bias_variable([1])
    h_d_deconv8 = tf.nn.conv2d_transpose(h_d_conv7, W_d_conv8, output_shape=[batch_size, 14, 28, 32], strides=strides, padding='SAME')
    h_d_conv8 = tf.nn.relu(h_d_deconv8 + b_d_conv8)

    # layer 9
    W_d_conv9 = weight_variable([3, 3, 32, 32])
    b_d_conv9 = bias_variable([1])
    h_d_deconv9 = tf.nn.conv2d_transpose(h_d_conv8, W_d_conv9, output_shape=[batch_size, 14, 28, 32], strides=strides, padding='SAME')
    h_d_conv9 = tf.nn.relu(h_d_deconv9 + b_d_conv9)

    # layer 10
    W_d_conv10 = weight_variable([3, 3, 1, 32])
    b_d_conv10 = bias_variable([1])
    h_d_deconv10 = tf.nn.conv2d_transpose(h_d_conv9, W_d_conv10, output_shape=[batch_size, 14, 28, 1], strides=strides, padding='SAME')
    h_d_conv10 = tf.nn.relu(h_d_deconv10 + b_d_conv10)

    y_image = tf.reshape(h_d_conv10, [-1, 392])

    y_conv = y_image

    cost = tf.reduce_mean(tf.mul(y_conv - y_, y_conv - y_)) #+ 0.0001*tf.reduce_sum(tf.abs(W_fc2))
    cost_summary = tf.scalar_summary('cost', cost)
    mse = tf.reduce_mean(tf.mul(y_conv - y_, y_conv - y_))
    mse_summary = tf.scalar_summary('mse', mse)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

  return Net()

def create_network_autoencoder(maskVecXoneYzero, bottleneck, learning_rate=1e-3):
  class Net:
    epsilon = 1e-3      # define small epsilon for batch normalization
    keep_prob = tf.placeholder(tf.float32)      # keep prob for dropout

    # layer 0
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1,28,28,1]) # [-1,14,28,1]
    
    # ---- encoder ----
    # layer 1
    W_e_conv1 = weight_variable([5, 5, 1, 32])
    b_e_conv1 = bias_variable([32])
    h_e_conv1 = tf.nn.relu(conv2d(x_image, W_e_conv1) + b_e_conv1)
    
    h_pool1 = max_pool_2x2(h_e_conv1)

    # layer 2
    W_e_conv2 = weight_variable([5, 5, 32, 64])
    b_e_conv2 = bias_variable([64])
    h_e_conv2 = tf.nn.relu(conv2d(h_pool1, W_e_conv2) + b_e_conv2)

    h_pool2 = max_pool_2x2(h_e_conv2)

    # ---- fully connected layer ----
    dim_fc_in =  7 * 7 * 64 #4 * 7 * 64
    W_e_fc1 = weight_variable([dim_fc_in, bottleneck])
    b_e_fc1 = bias_variable([bottleneck])

    h_e_convs_flat = tf.reshape(h_pool2, [-1, dim_fc_in])
    h_e_fc1 = tf.nn.relu(tf.matmul(h_e_convs_flat, W_e_fc1) + b_e_fc1)

    # with drop out
    keep_prob = tf.placeholder(tf.float32)
    h_e_fc1_drop = tf.nn.dropout(h_e_fc1, keep_prob)

    # decoder input
    dim_fc_out = 7 * 7 * 64  #4 * 7 * 64
    W_d_fc1 = weight_variable([bottleneck, dim_fc_out])
    b_d_fc1 = bias_variable([dim_fc_out])
    h_d_fc1 = tf.nn.relu(tf.matmul(h_e_fc1_drop, W_d_fc1) + b_d_fc1)
    h_d_fc1_drop = tf.nn.dropout(h_d_fc1, keep_prob)

    decoder_input = tf.reshape(h_d_fc1_drop, [-1, 7, 7, 64]) #tf.reshape(h_d_fc1_drop, [-1, 4, 7, 64])

    # ---- decoder ----
    strides=[1, 2, 2, 1]
    batch_size = tf.shape(x)[0]

    # layer 2
    W_d_conv2 = weight_variable([5, 5, 32, 64])
    b_d_conv2 = bias_variable([32])
    h_d_deconv2 = tf.nn.conv2d_transpose(decoder_input, W_d_conv2, output_shape=[batch_size, 14, 14, 32], strides=strides, padding='SAME')
    h_d_conv2 = tf.nn.relu(h_d_deconv2 + b_d_conv2)

    # layer 1
    W_d_conv1 = weight_variable([5, 5, 1, 32])
    b_d_conv1 = bias_variable([1])
    h_d_deconv1 = tf.nn.conv2d_transpose(h_d_conv2, W_d_conv1, output_shape=[batch_size, 28, 28, 1], strides=strides, padding='SAME')
    h_d_conv1 = tf.nn.relu(h_d_deconv1 + b_d_conv1)
    
    y_conv = tf.reshape(h_d_conv1, [-1, 784])

    # this is now [784, batch_size]
    y_conv_transpose = tf.matrix_transpose(y_conv)
    y_transpose = tf.matrix_transpose(y)

    maskVec = np.reshape(maskVecXoneYzero, 784)
    maskVecBool = np.ones((784)) == (np.ones((784)) - maskVec)
    y_conv_masked = tf.boolean_mask(y_conv_transpose, maskVecBool) 
    y_masked = tf.boolean_mask(y_transpose, maskVecBool)

    #print np.sum(maskVecBool)
    #print y_conv_transpose.get_shape()
    #print y_transpose.get_shape()
    #print y_conv_masked.get_shape()
    #print y_masked.get_shape()
    '''
    # ---- decoder ----
    strides=[1, 2, 2, 1]
    batch_size = tf.shape(x)[0]

    # layer 2
    W_d_conv2 = weight_variable([5, 5, 32, 64])
    b_d_conv2 = bias_variable([32])
    h_d_deconv2 = tf.nn.conv2d_transpose(decoder_input, W_d_conv2, output_shape=[batch_size, 7, 14, 32], strides=strides, padding='SAME')
    h_d_conv2 = tf.nn.relu(h_d_deconv2 + b_d_conv2)

    # layer 1
    W_d_conv1 = weight_variable([5, 5, 1, 32])
    b_d_conv1 = bias_variable([1])
    h_d_deconv1 = tf.nn.conv2d_transpose(h_d_conv2, W_d_conv1, output_shape=[batch_size, 14, 28, 1], strides=strides, padding='SAME')
    h_d_conv1 = tf.nn.relu(h_d_deconv1 + b_d_conv1)
    
    y_conv = tf.reshape(h_d_conv1, [-1, 392])
    '''

    #cost = tf.reduce_mean(tf.mul(y_conv - y_, y_conv - y_)) #+ 0.0001*tf.reduce_sum(tf.abs(W_fc2))
    #cost = tf.reduce_mean(tf.mul(y_conv - y, y_conv - y))
    cost = tf.reduce_mean(tf.mul(y_conv_masked - y_masked, y_conv_masked - y_masked))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

  return Net()