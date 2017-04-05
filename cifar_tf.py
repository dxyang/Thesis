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

def conv2d_stride2(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def leaky_relu(x, leak=0.1):
  return tf.maximum(x, leak*x)


def create_autoencoder(maskVecXoneYzero, bottleneck, learning_rate=1e-3):
  class Net:
    epsilon = 1e-3      # define small epsilon for batch normalization
    keep_prob = tf.placeholder(tf.float32)      # keep prob for dropout

    # layer 0
    x = tf.placeholder(tf.float32, shape=[None, 3072])
    y = tf.placeholder(tf.float32, shape=[None, 3072])
    x_image_pre = tf.reshape(x, [-1, 3, 32, 32])
    x_image = tf.transpose(x_image_pre, perm=[0, 2, 3, 1])

    # ---- encoder ----
    
    # batch_mean1, batch_var1 = tf.nn.moments(h_e_conv1_pre, axes=[0, 1, 2])
    # scale1 = tf.Variable(tf.ones([64]))
    # beta1 = tf.Variable(tf.zeros([64]))
    # bn1 = tf.nn.batch_normalization(h_e_conv1_pre, batch_mean1, batch_var1, beta1, scale1, epsilon)

    # batch_mean2, batch_var2 = tf.nn.moments(h_e_conv2_pre, axes=[0, 1, 2])
    # scale2 = tf.Variable(tf.ones([128]))
    # beta2 = tf.Variable(tf.zeros([128]))
    # bn2 = tf.nn.batch_normalization(h_e_conv2_pre, batch_mean2, batch_var2, beta2, scale2, epsilon)

    # batch_mean3, batch_var3 = tf.nn.moments(h_e_conv3_pre, axes=[0, 1, 2])
    # scale3 = tf.Variable(tf.ones([256]))
    # beta3 = tf.Variable(tf.zeros([256]))
    # bn3 = tf.nn.batch_normalization(h_e_conv3_pre, batch_mean3, batch_var3, beta3, scale3, epsilon)

    # layer 1 (conv - bn - relu - pool)
    W_e_conv1 = weight_variable([3, 3, 3, 32])
    b_e_conv1 = bias_variable([32])
    h_e_conv1_pre = conv2d(x_image, W_e_conv1)
    h_e_conv1 = tf.nn.relu(h_e_conv1_pre + b_e_conv1)

    W_e_conv1_b = weight_variable([3, 3, 32, 32])
    b_e_conv1_b = bias_variable([32])
    h_e_conv1_b_pre = conv2d(h_e_conv1, W_e_conv1_b)
    h_e_conv1_b = tf.nn.relu(h_e_conv1_b_pre + b_e_conv1_b)

    h_pool1 = max_pool_2x2(h_e_conv1_b)

    # layer 2 (conv - bn - relu - pool)
    W_e_conv2 = weight_variable([3, 3, 32, 64])
    b_e_conv2 = bias_variable([64])
    h_e_conv2_pre = conv2d(h_pool1, W_e_conv2)
    h_e_conv2 = tf.nn.relu(h_e_conv2_pre + b_e_conv2)

    W_e_conv2_b = weight_variable([3, 3, 64, 64])
    b_e_conv2_b = bias_variable([64])
    h_e_conv2_b_pre = conv2d(h_e_conv2, W_e_conv2_b)
    h_e_conv2_b = tf.nn.relu(h_e_conv2_b_pre + b_e_conv2_b)

    h_pool2 = max_pool_2x2(h_e_conv2_b)

    # layer 3 (conv - bn - relu - pool)
    W_e_conv3 = weight_variable([3, 3, 64, 128])
    b_e_conv3 = bias_variable([128])
    h_e_conv3_pre = conv2d(h_pool2, W_e_conv3)
    h_e_conv3 = tf.nn.relu(h_e_conv3_pre + b_e_conv3)

    W_e_conv3_b = weight_variable([3, 3, 128, 128])
    b_e_conv3_b = bias_variable([128])
    h_e_conv3_b_pre = conv2d(h_e_conv3, W_e_conv3_b)
    h_e_conv3_b = tf.nn.relu(h_e_conv3_b_pre + b_e_conv3_b)

    h_pool3 = max_pool_2x2(h_e_conv3_b)

    # ---- fully connected layers ----
    dim_fc_in =  4 * 4 * 128
    W_e_fc1 = weight_variable([dim_fc_in, 1024])
    b_e_fc1 = bias_variable([1024])

    h_e_convs_flat = tf.reshape(h_pool3, [-1, dim_fc_in])
    h_e_fc1 = tf.nn.relu(tf.matmul(h_e_convs_flat, W_e_fc1) + b_e_fc1)

    h_e_fc1_drop = tf.nn.dropout(h_e_fc1, keep_prob)

    W_e_fc2 = weight_variable([1024, bottleneck])
    b_e_fc2 = bias_variable([bottleneck])
    h_e_fc2 = tf.nn.relu(tf.matmul(h_e_fc1, W_e_fc2) + b_e_fc2)

    h_e_fc2_drop = tf.nn.dropout(h_e_fc2, keep_prob)

    # ---- decoder ----
    # ---- fully connected layers ----
    W_d_fc2 = weight_variable([bottleneck, 1024])
    b_d_fc2 = bias_variable([1024])
    h_d_fc2 = tf.nn.relu(tf.matmul(h_e_fc2, W_d_fc2) + b_d_fc2)
    h_d_fc2_drop = tf.nn.dropout(h_d_fc2, keep_prob)

    dim_fc_out =  4 * 4 * 128
    W_d_fc1 = weight_variable([1024, dim_fc_out])
    b_d_fc1 = bias_variable([dim_fc_out])
    h_d_fc1 = tf.nn.relu(tf.matmul(h_d_fc2, W_d_fc1) + b_d_fc1)
    h_d_fc1_drop = tf.nn.dropout(h_d_fc1, keep_prob)

    decoder_input = tf.reshape(h_d_fc1, [-1, 4, 4, 128])

    # ---- convolutional layers ----
    strides=[1, 2, 2, 1]
    strides_same=[1, 1, 1, 1]
    batch_size = tf.shape(x)[0]

    # layer 3
    W_d_conv3_b = weight_variable([3, 3, 128, 128])
    b_d_conv3_b = bias_variable([128])
    h_d_deconv3_b = tf.nn.conv2d_transpose(decoder_input, W_d_conv3_b, output_shape=[batch_size, 4, 4, 128], strides=strides_same, padding='SAME')
    h_d_conv3_b = tf.nn.relu(h_d_deconv3_b + b_d_conv3_b)

    W_d_conv3 = weight_variable([3, 3, 64, 128])
    b_d_conv3 = bias_variable([64])
    h_d_deconv3 = tf.nn.conv2d_transpose(h_d_conv3_b, W_d_conv3, output_shape=[batch_size, 8, 8, 64], strides=strides, padding='SAME')
    h_d_conv3 = tf.nn.relu(h_d_deconv3 + b_d_conv3)

    # layer 2
    W_d_conv2_b = weight_variable([3, 3, 64, 64])
    b_d_conv2_b = bias_variable([64])
    h_d_deconv2_b = tf.nn.conv2d_transpose(h_d_conv3, W_d_conv2_b, output_shape=[batch_size, 8, 8, 64], strides=strides_same, padding='SAME')
    h_d_conv2_b = tf.nn.relu(h_d_deconv2_b + b_d_conv2_b)

    W_d_conv2 = weight_variable([3, 3, 32, 64])
    b_d_conv2 = bias_variable([32])
    h_d_deconv2 = tf.nn.conv2d_transpose(h_d_conv2_b, W_d_conv2, output_shape=[batch_size, 16, 16, 32], strides=strides, padding='SAME')
    h_d_conv2 = tf.nn.relu(h_d_deconv2 + b_d_conv2)

    # layer 1
    W_d_conv1_b = weight_variable([3, 3, 32, 32])
    b_d_conv1_b = bias_variable([32])
    h_d_deconv1_b = tf.nn.conv2d_transpose(h_d_conv2, W_d_conv1_b, output_shape=[batch_size, 16, 16, 32], strides=strides_same, padding='SAME')
    h_d_conv1_b = tf.nn.relu(h_d_deconv1_b + b_d_conv1_b)   

    W_d_conv1 = weight_variable([3, 3, 3, 32])
    b_d_conv1 = bias_variable([3])
    h_d_deconv1 = tf.nn.conv2d_transpose(h_d_conv1_b, W_d_conv1, output_shape=[batch_size, 32, 32, 3], strides=strides, padding='SAME')
    h_d_conv1 = tf.nn.relu(h_d_deconv1 + b_d_conv1)   

    #h_d_conv1_rearrange = tf.transpose(h_d_conv1, perm=[0, 3, 1, 2])

    y_conv = tf.reshape(h_d_conv1, [-1, 3072])
    
    '''
    cost = tf.reduce_mean(tf.mul(y_conv - y, y_conv - y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
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

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

  return Net()

def create_autoencoder_adversarial(bottleneck, learning_rate=1e-3):
  class Net:
    epsilon = 1e-3      # define small epsilon for batch normalization
    keep_prob = tf.placeholder(tf.float32)      # keep prob for dropout

    # layer 0
    x = tf.placeholder(tf.float32, shape=[None, 3072])
    y = tf.placeholder(tf.float32, shape=[None, 768])
    x_image_pre = tf.reshape(x, [-1, 3, 32, 32])
    x_image = tf.transpose(x_image_pre, perm=[0, 2, 3, 1])
    y_image_pre = tf.reshape(y, [-1, 3, 16, 16])
    y_image = tf.transpose(y_image_pre, perm=[0, 2, 3, 1])

    # ---- encoder ----
    
    # batch_mean1, batch_var1 = tf.nn.moments(h_e_conv1_pre, axes=[0, 1, 2])
    # scale1 = tf.Variable(tf.ones([64]))
    # beta1 = tf.Variable(tf.zeros([64]))
    # bn1 = tf.nn.batch_normalization(h_e_conv1_pre, batch_mean1, batch_var1, beta1, scale1, epsilon)

    # batch_mean2, batch_var2 = tf.nn.moments(h_e_conv2_pre, axes=[0, 1, 2])
    # scale2 = tf.Variable(tf.ones([128]))
    # beta2 = tf.Variable(tf.zeros([128]))
    # bn2 = tf.nn.batch_normalization(h_e_conv2_pre, batch_mean2, batch_var2, beta2, scale2, epsilon)

    # batch_mean3, batch_var3 = tf.nn.moments(h_e_conv3_pre, axes=[0, 1, 2])
    # scale3 = tf.Variable(tf.ones([256]))
    # beta3 = tf.Variable(tf.zeros([256]))
    # bn3 = tf.nn.batch_normalization(h_e_conv3_pre, batch_mean3, batch_var3, beta3, scale3, epsilon)

    # layer 1 (conv - bn - relu - pool)
    W_e_conv1 = weight_variable([3, 3, 3, 32])
    b_e_conv1 = bias_variable([32])
    h_e_conv1_pre = conv2d(x_image, W_e_conv1)
    h_e_conv1 = tf.nn.relu(h_e_conv1_pre + b_e_conv1)

    W_e_conv1_b = weight_variable([3, 3, 32, 32])
    b_e_conv1_b = bias_variable([32])
    h_e_conv1_b_pre = conv2d(h_e_conv1, W_e_conv1_b)
    h_e_conv1_b = tf.nn.relu(h_e_conv1_b_pre + b_e_conv1_b)

    h_pool1 = max_pool_2x2(h_e_conv1_b)

    # layer 2 (conv - bn - relu - pool)
    W_e_conv2 = weight_variable([3, 3, 32, 64])
    b_e_conv2 = bias_variable([64])
    h_e_conv2_pre = conv2d(h_pool1, W_e_conv2)
    h_e_conv2 = tf.nn.relu(h_e_conv2_pre + b_e_conv2)

    W_e_conv2_b = weight_variable([3, 3, 64, 64])
    b_e_conv2_b = bias_variable([64])
    h_e_conv2_b_pre = conv2d(h_e_conv2, W_e_conv2_b)
    h_e_conv2_b = tf.nn.relu(h_e_conv2_b_pre + b_e_conv2_b)

    h_pool2 = max_pool_2x2(h_e_conv2_b)

    # layer 3 (conv - bn - relu - pool)
    W_e_conv3 = weight_variable([3, 3, 64, 128])
    b_e_conv3 = bias_variable([128])
    h_e_conv3_pre = conv2d(h_pool2, W_e_conv3)
    h_e_conv3 = tf.nn.relu(h_e_conv3_pre + b_e_conv3)

    W_e_conv3_b = weight_variable([3, 3, 128, 128])
    b_e_conv3_b = bias_variable([128])
    h_e_conv3_b_pre = conv2d(h_e_conv3, W_e_conv3_b)
    h_e_conv3_b = tf.nn.relu(h_e_conv3_b_pre + b_e_conv3_b)

    h_pool3 = max_pool_2x2(h_e_conv3_b)

    # ---- fully connected layers ----
    dim_fc_in =  4 * 4 * 128
    W_e_fc1 = weight_variable([dim_fc_in, 1024])
    b_e_fc1 = bias_variable([1024])

    h_e_convs_flat = tf.reshape(h_pool3, [-1, dim_fc_in])
    h_e_fc1 = tf.nn.relu(tf.matmul(h_e_convs_flat, W_e_fc1) + b_e_fc1)

    h_e_fc1_drop = tf.nn.dropout(h_e_fc1, keep_prob)

    W_e_fc2 = weight_variable([1024, bottleneck])
    b_e_fc2 = bias_variable([bottleneck])
    h_e_fc2 = tf.nn.relu(tf.matmul(h_e_fc1, W_e_fc2) + b_e_fc2)

    h_e_fc2_drop = tf.nn.dropout(h_e_fc2, keep_prob)

    # ---- decoder ----
    # ---- fully connected layers ----
    W_d_fc2 = weight_variable([bottleneck, 1024])
    b_d_fc2 = bias_variable([1024])
    h_d_fc2 = tf.nn.relu(tf.matmul(h_e_fc2, W_d_fc2) + b_d_fc2)
    h_d_fc2_drop = tf.nn.dropout(h_d_fc2, keep_prob)

    dim_fc_out =  4 * 4 * 128
    W_d_fc1 = weight_variable([1024, dim_fc_out])
    b_d_fc1 = bias_variable([dim_fc_out])
    h_d_fc1 = tf.nn.relu(tf.matmul(h_d_fc2, W_d_fc1) + b_d_fc1)
    h_d_fc1_drop = tf.nn.dropout(h_d_fc1, keep_prob)

    decoder_input = tf.reshape(h_d_fc1, [-1, 4, 4, 128])

    # ---- convolutional layers ----
    strides=[1, 2, 2, 1]
    strides_same=[1, 1, 1, 1]
    batch_size = tf.shape(x)[0]

    # layer 3
    W_d_conv3_b = weight_variable([3, 3, 128, 128])
    b_d_conv3_b = bias_variable([128])
    h_d_deconv3_b = tf.nn.conv2d_transpose(decoder_input, W_d_conv3_b, output_shape=[batch_size, 4, 4, 128], strides=strides_same, padding='SAME')
    h_d_conv3_b = tf.nn.relu(h_d_deconv3_b + b_d_conv3_b)

    W_d_conv3 = weight_variable([3, 3, 64, 128])
    b_d_conv3 = bias_variable([64])
    h_d_deconv3 = tf.nn.conv2d_transpose(h_d_conv3_b, W_d_conv3, output_shape=[batch_size, 8, 8, 64], strides=strides, padding='SAME')
    h_d_conv3 = tf.nn.relu(h_d_deconv3 + b_d_conv3)

    # layer 2
    W_d_conv2_b = weight_variable([3, 3, 64, 64])
    b_d_conv2_b = bias_variable([64])
    h_d_deconv2_b = tf.nn.conv2d_transpose(h_d_conv3, W_d_conv2_b, output_shape=[batch_size, 8, 8, 64], strides=strides_same, padding='SAME')
    h_d_conv2_b = tf.nn.relu(h_d_deconv2_b + b_d_conv2_b)

    W_d_conv2 = weight_variable([3, 3, 32, 64])
    b_d_conv2 = bias_variable([32])
    h_d_deconv2 = tf.nn.conv2d_transpose(h_d_conv2_b, W_d_conv2, output_shape=[batch_size, 16, 16, 32], strides=strides, padding='SAME')
    h_d_conv2 = tf.nn.relu(h_d_deconv2 + b_d_conv2)

    # layer 1
    W_d_conv1_b = weight_variable([3, 3, 3, 32])
    b_d_conv1_b = bias_variable([3])
    h_d_deconv1_b = tf.nn.conv2d_transpose(h_d_conv2, W_d_conv1_b, output_shape=[batch_size, 16, 16, 3], strides=strides_same, padding='SAME')
    h_d_conv1_b = tf.nn.relu(h_d_deconv1_b + b_d_conv1_b)   

    #h_d_conv1_rearrange = tf.transpose(h_d_conv1, perm=[0, 3, 1, 2])

    y_conv = tf.reshape(h_d_conv1_b, [-1, 768])

    #l2 loss
    cost = tf.reduce_mean(tf.mul(y_conv - y, y_conv - y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    g_vars = [W_e_conv1, b_e_conv1, W_e_conv1_b, b_e_conv1_b, 
                W_e_conv2, b_e_conv2, W_e_conv2_b, b_e_conv2_b,
                W_e_conv3, b_e_conv3, W_e_conv3_b, b_e_conv3_b,
                W_e_fc1, b_e_fc1, W_e_fc2, b_e_fc2,
                W_d_fc2, b_d_fc2, W_d_fc1, b_d_fc1,
                W_d_conv3_b, b_d_conv3_b, W_d_conv3, b_d_conv3,
                W_d_conv2_b, b_d_conv2_b, W_d_conv2, b_d_conv2,
                W_d_conv1_b, b_d_conv1_b]

    # ---- discriminator variables ----
    global W_disc_conv1; W_disc_conv1 = weight_variable([3, 3, 3, 64])
    global b_disc_conv1; b_disc_conv1 = bias_variable([64])
    global W_disc_conv1_b; W_disc_conv1_b = weight_variable([3, 3, 64, 64])
    global b_disc_conv1_b; b_disc_conv1_b = bias_variable([64])

    global W_disc_conv2; W_disc_conv2 = weight_variable([3, 3, 64, 128])
    global b_disc_conv2; b_disc_conv2 = bias_variable([128])
    global W_disc_conv2_b; W_disc_conv2_b = weight_variable([3, 3, 128, 128])
    global b_disc_conv2_b; b_disc_conv2_b = bias_variable([128])

    global dim_fc_discriminator; dim_fc_discriminator = 4*4*128
    global W_disc_fc1; W_disc_fc1 = weight_variable([dim_fc_discriminator, 1024])
    global b_disc_fc1; b_disc_fc1 = bias_variable([1024])

    global W_discriminate; W_discriminate = weight_variable([1024, 1])
    global b_discriminate; b_discriminate = bias_variable([1])

    d_vars = [W_disc_conv1, b_disc_conv1, W_disc_conv1_b, b_disc_conv1_b, 
                W_disc_conv2, b_disc_conv2, W_disc_conv2_b, b_disc_conv2_b,
                W_disc_fc1, b_disc_fc1, W_discriminate, b_discriminate]


    # ----- DISCRIMINATOR ADVERSARIAL STUFF ---- 
    def discriminator(input):
        # ---- convolutional layer 1 ----
        h_disc_conv1_pre = conv2d(input, W_disc_conv1)
        h_disc_conv1 = leaky_relu(h_disc_conv1_pre + b_disc_conv1)
        h_disc_conv1_b_pre = conv2d(h_disc_conv1, W_disc_conv1_b)
        h_disc_conv1_b = leaky_relu(h_disc_conv1_b_pre + b_disc_conv1_b)

        h_disc_pool_1 = max_pool_2x2(h_disc_conv1_b)

        # ---- convolutional layer 2 ----
        h_disc_conv2_pre = conv2d(h_disc_pool_1, W_disc_conv2)
        h_disc_conv2 = leaky_relu(h_disc_conv2_pre + b_disc_conv2)
        h_disc_conv2_b_pre = conv2d(h_disc_conv2, W_disc_conv2_b)
        h_disc_conv2_b = leaky_relu(h_disc_conv2_b_pre + b_disc_conv2_b)

        h_disc_pool2 = max_pool_2x2(h_disc_conv2_b)

        h_disc_conv2_flat = tf.reshape(h_disc_pool2, [-1, dim_fc_discriminator])

        # ---- fully connected layer 1 ----
        h_disc_fc1 = leaky_relu(tf.matmul(h_disc_conv2_flat, W_disc_fc1) + b_disc_fc1)
        h_disc_fc1_drop = tf.nn.dropout(h_disc_fc1, 0.5)

        # ---- discrimnate ----
        d_discriminate = tf.sigmoid(tf.matmul(h_disc_fc1_drop, W_discriminate) + b_discriminate)
        
        return d_discriminate

    discriminator_fake = discriminator(h_d_conv1_b)     # D(F((1-M)*x))
    discriminator_real = discriminator(y_image)       # D(x)

    avg_disc_fake = tf.reduce_mean(discriminator_fake)
    avg_disc_real = tf.reduce_mean(discriminator_real)

    d_loss = -tf.reduce_mean(tf.log(discriminator_real + 1e-10) + tf.log(1.0 - discriminator_fake + 1e-10))
    g_loss = -tf.reduce_mean(tf.log(discriminator_fake + 1e-10))

    total_cost = 0.9*cost + 0.1*g_loss

    d_step = tf.train.AdamOptimizer(2e-4).minimize(d_loss, var_list=d_vars)
    g_step = tf.train.AdamOptimizer(1e-3).minimize(total_cost, var_list=g_vars)

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

  return Net()


def create_autoencoder_classification(bottleneck, learning_rate=1e-3):
  class Net:
    epsilon = 1e-3      # define small epsilon for batch normalization
    keep_prob = tf.placeholder(tf.float32)      # keep prob for dropout

    # layer 0
    x = tf.placeholder(tf.float32, shape=[None, 3072])
    labels = tf.placeholder(tf.float32, shape=[None])
    labels = tf.cast(labels, tf.int64)
    x_image_pre = tf.reshape(x, [-1, 3, 32, 32])
    x_image = tf.transpose(x_image_pre, perm=[0, 2, 3, 1])

    # ---- encoder ----

    # batch_mean2, batch_var2 = tf.nn.moments(h_e_conv2_pre, axes=[0, 1, 2])
    # scale2 = tf.Variable(tf.ones([128]))
    # beta2 = tf.Variable(tf.zeros([128]))
    # bn2 = tf.nn.batch_normalization(h_e_conv2_pre, batch_mean2, batch_var2, beta2, scale2, epsilon)

    # batch_mean3, batch_var3 = tf.nn.moments(h_e_conv3_pre, axes=[0, 1, 2])
    # scale3 = tf.Variable(tf.ones([256]))
    # beta3 = tf.Variable(tf.zeros([256]))
    # bn3 = tf.nn.batch_normalization(h_e_conv3_pre, batch_mean3, batch_var3, beta3, scale3, epsilon)

    # layer set 1 
    W_e_conv1 = weight_variable([3, 3, 3, 32])
    b_e_conv1 = bias_variable([32])
    h_e_conv1_pre = conv2d(x_image, W_e_conv1)
    # batch_mean1, batch_var1 = tf.nn.moments(h_e_conv1_pre, axes=[0, 1, 2])
    # scale1 = tf.Variable(tf.ones([32]))
    # beta1 = tf.Variable(tf.zeros([32]))
    # bn1 = tf.nn.batch_normalization(h_e_conv1_pre, batch_mean1, batch_var1, beta1, scale1, epsilon)
    h_e_conv1 = tf.nn.relu(h_e_conv1_pre + b_e_conv1)

    W_e_conv1_b = weight_variable([3, 3, 32, 32])
    b_e_conv1_b = bias_variable([32])
    h_e_conv1_b_pre = conv2d(h_e_conv1, W_e_conv1_b)
    h_e_conv1_b = tf.nn.relu(h_e_conv1_b_pre + b_e_conv1_b)

    h_pool1 = max_pool_2x2(h_e_conv1_b)

    # layer 2
    W_e_conv2 = weight_variable([3, 3, 32, 64])
    b_e_conv2 = bias_variable([64])
    h_e_conv2_pre = conv2d(h_pool1, W_e_conv2)
    h_e_conv2 = tf.nn.relu(h_e_conv2_pre + b_e_conv2)

    W_e_conv2_b = weight_variable([3, 3, 64, 64])
    b_e_conv2_b = bias_variable([64])
    h_e_conv2_b_pre = conv2d(h_e_conv2, W_e_conv2_b)
    h_e_conv2_b = tf.nn.relu(h_e_conv2_b_pre + b_e_conv2_b)

    h_pool2 = max_pool_2x2(h_e_conv2_b)

    # layer 3 (conv - relu)
    W_e_conv3 = weight_variable([3, 3, 64, 128])
    b_e_conv3 = bias_variable([128])
    h_e_conv3_pre = conv2d(h_pool2, W_e_conv3)
    h_e_conv3 = tf.nn.relu(h_e_conv3_pre + b_e_conv3)

    W_e_conv3_b = weight_variable([3, 3, 128, 128])
    b_e_conv3_b = bias_variable([128])
    h_e_conv3_b_pre = conv2d(h_e_conv3, W_e_conv3_b)
    h_e_conv3_b = tf.nn.relu(h_e_conv3_b_pre + b_e_conv3_b)

    h_pool3 = max_pool_2x2(h_e_conv3_b)

    # ---- fully connected layers ----

    dim_fc_in =  4 * 4 * 128
    W_e_fc1 = weight_variable([dim_fc_in, 1024])
    b_e_fc1 = bias_variable([1024])

    h_e_convs_flat = tf.reshape(h_pool3, [-1, dim_fc_in])
    h_e_fc1 = tf.nn.relu(tf.matmul(h_e_convs_flat, W_e_fc1) + b_e_fc1)

    #h_e_fc1_drop = tf.nn.dropout(h_e_fc1, 0.7)

    W_e_fc2 = weight_variable([1024, bottleneck])
    b_e_fc2 = bias_variable([bottleneck])
    h_e_fc2 = tf.nn.relu(tf.matmul(h_e_fc1, W_e_fc2) + b_e_fc2)

    #h_e_fc2_drop = tf.nn.dropout(h_e_fc2, keep_prob)

    W = weight_variable([bottleneck, 10])
    b = bias_variable([10])
    h = (tf.matmul(h_e_fc2, W) + b)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=h)
    cost = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(labels, tf.argmax(h,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

  return Net()