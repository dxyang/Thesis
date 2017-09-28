import tensorflow as tf
import numpy as np
import os
import math
import mnist_preprocessing
import time

# --- import mnist from tf (adapted from deepmnist tutorial) ----
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_mnist, test_mnist = mnist_preprocessing.returnData()

train_hideMiddle, Xtrain_hideMiddle, Ytrain_hideMiddle, \
test_hideMiddle, Xtest_hideMiddle, Ytest_hideMiddle = mnist_preprocessing.returnSquareData(squareSideLength=14)

# train_autoencoder, test_autoencoder = mnist_preprocessing.returnAutoencoderData()

# ---- helper functions (adapted from deepmnist tutorial) ----
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def leaky_relu(x, leak=0.1):
  return tf.maximum(x, leak*x)

###################################
#   GAN VARIABLES
###################################

# ---- generator variables ----
W_g_fc2 = weight_variable([62, 1024])
b_g_fc2 = bias_variable([1024])

dim_fc_out = 7*7*128
W_g_fc1 = weight_variable([1024, dim_fc_out])
b_g_fc1 = bias_variable([dim_fc_out])
g_fc_scale1 = tf.Variable(tf.ones([dim_fc_out]))
g_fc_beta1 = tf.Variable(tf.zeros([dim_fc_out]))

W_g_upconv2 = weight_variable([4, 4, 64, 128])
b_g_upconv2 = bias_variable([64])
g_scale2 = tf.Variable(tf.ones([64]))
g_beta2 = tf.Variable(tf.zeros([64]))

W_g_upconv1 = weight_variable([4, 4, 1, 64])
b_g_upconv1 = bias_variable([1])
g_scale1 = tf.Variable(tf.ones([1]))
g_beta1 = tf.Variable(tf.zeros([1]))

# ---- discriminator variables ----
W_d_conv1 = weight_variable([4, 4, 1, 64])
b_d_conv1 = bias_variable([64])
#d_scale1 = tf.Variable(tf.ones([64]))
#d_beta1 = tf.Variable(tf.zeros([64]))

W_d_conv2 = weight_variable([4, 4, 64, 128])
b_d_conv2 = bias_variable([128])
#d_scale2 = tf.Variable(tf.ones([128]))
#d_beta2 = tf.Variable(tf.zeros([128]))

dim_fc_in = 7*7*128
W_d_fc1 = weight_variable([dim_fc_in, 1024])
b_d_fc1 = bias_variable([1024])
#d_fc_scale1 = tf.Variable(tf.ones([1024]))
#d_fc_beta1 = tf.Variable(tf.zeros([1024]))

W_discriminate = weight_variable([1024, 1])
b_discriminate = bias_variable([1])

def generator(z):
    epsilon = 1e-3      # define small epsilon for batch normalization

    # ---- fully connected layer 2 ----
    h_g_fc2 = tf.nn.relu(tf.matmul(z, W_g_fc2) + b_g_fc2)

    # ---- fully connected layer 1 ----
    h_g_fc1_pre = tf.matmul(h_g_fc2, W_g_fc1)

    fc_mean1, fc_var1 = tf.nn.moments(h_g_fc1_pre,[0])
    h_g_bn_fc1 = tf.nn.batch_normalization(h_g_fc1_pre,fc_mean1,fc_var1,g_fc_beta1,g_fc_scale1,epsilon)

    h_g_fc1 = tf.nn.relu(h_g_bn_fc1 + b_g_fc1)
    h_g_fc1_reshape = tf.reshape(h_g_fc1, [-1, 7, 7, 128])

    # ---- upconvolutional layer 2 ----
    strides=[1, 2, 2, 1]
    batch_size = tf.shape(z)[0]

    h_g_upconv2 = tf.nn.conv2d_transpose(h_g_fc1_reshape, W_g_upconv2, output_shape=[batch_size, 14, 14, 64], strides=strides, padding='SAME')

    batch_mean2, batch_var2 = tf.nn.moments(h_g_upconv2, axes=[0, 1, 2])
    bn2 = tf.nn.batch_normalization(h_g_upconv2, batch_mean2, batch_var2, g_beta2, g_scale2, epsilon)

    h_g_uc2 = tf.nn.relu(bn2 + b_g_upconv2)

    # ---- upconvolutional layer 1 ----
    h_g_upconv1 = tf.nn.conv2d_transpose(h_g_uc2, W_g_upconv1, output_shape=[batch_size, 28, 28, 1], strides=strides, padding='SAME')

    batch_mean1, batch_var1 = tf.nn.moments(h_g_upconv1, axes=[0, 1, 2])
    bn1 = tf.nn.batch_normalization(h_g_upconv1, batch_mean1, batch_var1, g_beta1, g_scale1, epsilon)
    
    h_g_uc1 = tf.sigmoid(bn1 + b_g_upconv1)
    
    y = tf.reshape(h_g_uc1, [-1, 784])

    return y

def discriminator(input):
    dis_image = tf.reshape(input, [-1,28,28,1])
    epsilon = 1e-3      # define small epsilon for batch normalization

    # ---- convolutional layer 1 ----
    h_d_conv1_pre = conv2d(dis_image, W_d_conv1)

    #batch_mean1, batch_var1 = tf.nn.moments(h_d_conv1_pre, axes=[0, 1, 2])
    #bn1 = tf.nn.batch_normalization(h_d_conv1_pre, batch_mean1, batch_var1, d_beta1, d_scale1, epsilon)

    h_d_conv1 = leaky_relu(h_d_conv1_pre + b_d_conv1)

    # ---- convolutional layer 2 ----
    h_d_conv2_pre = conv2d(h_d_conv1, W_d_conv2)

    #batch_mean2, batch_var2 = tf.nn.moments(h_d_conv2_pre, axes=[0, 1, 2])
    #bn2 = tf.nn.batch_normalization(h_d_conv2_pre, batch_mean2, batch_var2, d_beta2, d_scale2, epsilon)

    h_d_conv2 = leaky_relu(h_d_conv2_pre + b_d_conv2)
    h_d_conv2_flat = tf.reshape(h_d_conv2, [-1, dim_fc_in])

    # ---- fully connected layer 1 ----
    h_d_fc1_pre = leaky_relu(tf.matmul(h_d_conv2_flat, W_d_fc1) + b_d_fc1)

    #fc_mean1, fc_var1 = tf.nn.moments(h_d_fc1_pre,[0])
    #fc_bn1 = tf.nn.batch_normalization(h_d_fc1_pre, fc_mean1, fc_var1, d_fc_beta1, d_fc_scale1, epsilon)

    h_d_fc1 = leaky_relu(h_d_fc1_pre + b_d_fc1)
    h_d_fc1_drop = tf.nn.dropout(h_d_fc1, 0.5)

    # ---- discrimnate ----
    d_discriminate = tf.sigmoid(tf.matmul(h_d_fc1_drop, W_discriminate) + b_discriminate)

    return d_discriminate

###################################
#   GAN TRAINING
###################################
realInput = tf.placeholder(tf.float32, shape=[None, 784])            # x
sampleNoise = tf.placeholder(tf.float32, shape=[None, 62])           # z

fakeInput = generator(sampleNoise)                  # G(z)
discriminator_fake = discriminator(fakeInput)       # D(G(z))
discriminator_real = discriminator(realInput)       # D(x)

d_loss = -tf.reduce_mean(tf.log(discriminator_real + 1e-10) + tf.log(1.0 - discriminator_fake + 1e-10))
g_loss = -tf.reduce_mean(tf.log(discriminator_fake + 1e-10))

d_vars = [W_d_conv1, b_d_conv1, W_d_conv2, b_d_conv2, 
          W_d_fc1, b_d_fc1, W_discriminate, b_discriminate]
g_vars = [W_g_fc2, b_g_fc2, W_g_fc1, b_g_fc1, g_fc_scale1, g_fc_beta1,
          W_g_upconv2, b_g_upconv2, g_beta2, g_scale2, W_g_upconv1, b_g_upconv1, g_beta1, g_scale1]

d_step = tf.train.AdamOptimizer(2e-4).minimize(d_loss, var_list=d_vars)
g_step = tf.train.AdamOptimizer(1e-3).minimize(g_loss, var_list=g_vars)

train_batch = 50
n_iterations = 40000

# Add ops to save and restore all the variables
saver = tf.train.Saver()

###################################
#   IMAGE COMPLETION
###################################
mask = tf.placeholder(tf.float32, shape=[None, 784])
inverseMask = tf.placeholder(tf.float32, shape=[None, 784])

# mix and match masks
hiddenImg_real = tf.mul(mask, realInput)
hiddenImg_fake = tf.mul(mask, fakeInput)
restorationSection = tf.mul(inverseMask, fakeInput)
reconstruction = tf.add(hiddenImg_real, restorationSection)

# contextual loss (L1 norm of visible pixels)
contextual_loss = tf.reduce_sum(
    tf.contrib.layers.flatten(
        tf.abs(hiddenImg_real - hiddenImg_fake)), 1)

# perceptual loss (generator loss)
perceptual_loss = g_loss #-tf.reduce_mean(tf.log(reconstruction + 1e-10)) #g_loss

# combine losses and compute gradient update
alpha = 1.0
total_loss = contextual_loss + alpha*perceptual_loss
grad_total_loss = tf.gradients(total_loss, sampleNoise)

# L2 loss for just the hidden section
boolMask = tf.placeholder(tf.bool, shape=[None, 784])
l2_real = tf.boolean_mask(realInput, boolMask) 
l2_fake = tf.boolean_mask(fakeInput, boolMask)
l2_loss = tf.reduce_mean(tf.mul(l2_real - l2_fake, l2_real - l2_fake))

# L1 loss
# contextual_loss = tf.reduce_sum(tf.abs(realInput - fakeInput))
# realInput_contextual = tf.mul(realInput, mask)
# fakeInput_contextual = tf.mul(fakeInput, mask)
# contextual_loss = tf.reduce_sum(tf.abs(realInput_contextual - fakeInput_contextual))

# L2 loss
# contextual_loss = tf.reduce_sum(tf.mul(realInput - fakeInput, realInput - fakeInput))
# realInput_masked_known = tf.boolean_mask(realInput, maskBool) 
# fakeInput_masked_known = tf.boolean_mask(fakeInput, maskBool)
# contextual_loss = tf.reduce_sum(tf.mul(realInput_masked_known - fakeInput_masked_known, realInput_masked_known - fakeInput_masked_known))

def train():
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        idxs = np.arange(55000)

        for i in range(n_iterations):
            # Get real images
            # Shuffle the training data every time we run through the set
            if (i%(55000/50) == 0):
                if (i != 0):
                    np.random.shuffle(idxs)
                bufferedIdxs = np.concatenate((idxs, idxs[0:50]))

            # Shift through the training set 50 items at a time
            leftIdx = (i*50)%55000
            rightIdx = leftIdx + 50
            mnist_batch = train_mnist[:, bufferedIdxs[leftIdx:rightIdx]].T

            # Sample noise
            noise = np.random.uniform(-1.0, 1.0, size=[train_batch, 62])
            #noise = np.random.normal(loc=0.0, scale=1.0, size=[train_batch, 62])

            if ((i % 100) == 0):
                dloss, gloss = sess.run([d_loss, g_loss], 
                                    feed_dict={realInput: mnist_batch, 
                                             sampleNoise: noise})
                print 'iteration: %d' %(i)
                print 'd loss: %f, g loss: %f' %(dloss, gloss)

                if (math.isnan(dloss) or math.isnan(gloss)):
                    print "Not training :("
                    break

                d_fake, d_real = sess.run([discriminator_fake, discriminator_real],
                            feed_dict={realInput: mnist_batch, 
                                     sampleNoise: noise})

                #print np.array(d_fake)
                #print np.array(d_real)

            d_step.run(feed_dict={realInput: mnist_batch, 
                                sampleNoise: noise})
            g_step.run(feed_dict={realInput: mnist_batch, 
                                sampleNoise: noise})


        # Save the model
        modelStr = "/tmp/gan_mnist.ckpt"
        save_path = saver.save(sess, os.getcwd() + modelStr)
        print("Model saved in file: %s" % save_path)

def predict():
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        restoreVars = True
        if (restoreVars):
            # Restore variables from disk.
            saver.restore(sess, os.getcwd() + "/tmp/gan_mnist.ckpt")
            print("Model restored.")

        # empty real input
        eri = np.zeros((1000, 784))

        # sample noise
        #noise = np.random.uniform(-1.0, 1.0, size=[50, 62])
        noise = np.random.normal(loc=0.0, scale=1.0, size=[1000, 62])

        testOutput = sess.run(fakeInput, feed_dict={realInput: eri, 
                                                  sampleNoise: noise})
        np.save('testOutput_gan_uni.npy', testOutput)

def image_completion():
    momentum = 0.9
    learningRate = 0.001
    batch_size = 100

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        restoreVars = True
        if (restoreVars):
            # Restore variables from disk.
            saver.restore(sess, os.getcwd() + "/tmp/gan_mnist.ckpt")
            print("Model restored.")

        # get the vector mask
        squareSideLength = 12
        vecMask_pre = mnist_preprocessing.generateCenterSquareMask(squareSideLength) # hidden part of the image is 0, known part is 1
        vecMask = np.zeros((batch_size, 784))
        for i in range(batch_size):
            vecMask[i] = vecMask_pre
        vecMaskMatrix = vecMask
        invVecMaskMatrix = np.ones((batch_size, 784)) - vecMaskMatrix
        boolMaskMatrix = np.ones((batch_size, 784)) == invVecMaskMatrix

        # perform projected gradient descent to find z that minimizes
        # total contextual and perceptual loss (i.e., argmin_z of loss)
        isTrain = True
        n_images = 0
        if (isTrain):
            n_images = 55000
        else:
            n_images = 10000

        avgLoss_array = np.zeros(n_images/batch_size)

        hiddenImgs = np.zeros((n_images, 784))
        generatedImgs = np.zeros((n_images, 784))
        reconstructedImgs = np.zeros((n_images, 784))
        bestNoiseVecs = np.zeros((n_images, 62))

        # restore from this checkpoint
        # tmpCkpt = np.load('tmp/bestNoiseVecs_10_10000_mnist_test.npy')

        start_time = time.time()
        for idx in range(n_images/batch_size):
            leftIdx = idx*batch_size
            rightIdx = (idx+1)*batch_size
            mnist_batch = 0
            if (isTrain):
                mnist_batch = train_mnist[:, leftIdx:rightIdx].T
            else:
                mnist_batch = test_mnist[:, leftIdx:rightIdx].T
            validationNum = 4
            val = 0

            # placeholders vars for this set of validation runs
            hiddenImgs_validate = np.zeros((validationNum, batch_size, 784))
            generatedImgs_validate = np.zeros((validationNum, batch_size, 784))
            reconstructedImgs_validate = np.zeros((validationNum, batch_size, 784))
            z_hats_validate = np.zeros((validationNum, batch_size, 62))
            l2_validate = np.ones((validationNum)) # this is init to 1!
            while (val < validationNum):
                # samples noise, start velocity at zero
                z_hats = np.random.uniform(-1.0, 1.0, size=[batch_size, 62])
                v = 0

                # restore from checkpoint
                gradDescentIterations = 2000
                # checkList = [6200, 6800, 7800, 7900, 8000, 8100] 
                # if (rightIdx not in checkList):
                #     print 'Idxs:[%5d:%5d] - Restoring Best Noise Vecs' \
                #             %(leftIdx, rightIdx)
                #     gradDescentIterations = 0
                #     val = validationNum - 1
                #     z_hats = tmpCkpt[leftIdx:rightIdx]

                
                # gradient descent
                for i in range(gradDescentIterations):
                    loss, gradient, con, percep, l2 = sess.run([total_loss, grad_total_loss, contextual_loss, perceptual_loss, l2_loss], 
                                                                feed_dict={realInput: mnist_batch, 
                                                                         sampleNoise: z_hats,
                                                                                mask: vecMaskMatrix,
                                                                         inverseMask: invVecMaskMatrix,
                                                                            boolMask: boolMaskMatrix})
                    # if (i % 1000) == 0:
                    #     avgLoss = np.mean(loss)
                    #     avgCon = np.mean(con)
                    #     print 'val %d idxs %d to %d, iteration %d, total: %g, contextual: %g, perceptual: %g, l2: %g' \
                    #             %(val, leftIdx, rightIdx, i, avgLoss, avgCon, percep, l2)

                    v_prev = np.copy(v)
                    v = momentum*v - learningRate*1.0*gradient[0]
                    z_hats += -momentum*v_prev + (1+momentum)*v
                    z_hats = np.clip(z_hats, -1, 1)

                # get hidden, generated, and reconstructed img as well as L2 loss for this run
                hiddenImg_batch, generatedImg_batch, reconstructedImg_batch, l2 = sess.run([hiddenImg_real, fakeInput, reconstruction, l2_loss], 
                                                                feed_dict={realInput: mnist_batch, 
                                                                         sampleNoise: z_hats,
                                                                                mask: vecMaskMatrix,
                                                                         inverseMask: invVecMaskMatrix,
                                                                            boolMask: boolMaskMatrix})
                print 'Idxs:[%5d:%5d] out of [%6d], Validation: [%2d], Time: %10.2f, L2: [%.6f]' \
                        %(leftIdx, rightIdx, n_images, val, time.time() - start_time, l2)

                # # L2 error too high; try resampling noise
                # if (l2 > 0.04) and (val == 4):
                #     print 'val %d l2 value too high, resampling noise...' %val
                #     continue

                # If L2 error not too high, keep this run to pick the min from later
                hiddenImgs_validate[val] = hiddenImg_batch
                generatedImgs_validate[val] = generatedImg_batch
                reconstructedImgs_validate[val] = reconstructedImg_batch
                l2_validate[val] = l2
                z_hats_validate[val] = z_hats
                val += 1

            # pick the best out of the validation runs
            bestMatchIdx = np.argmin(l2_validate)
            best_l2 = np.min(l2_validate)

            # store things!
            hiddenImgs[leftIdx:rightIdx, :] = hiddenImgs_validate[bestMatchIdx]
            generatedImgs[leftIdx:rightIdx, :] = generatedImgs_validate[bestMatchIdx]
            reconstructedImgs[leftIdx:rightIdx, :] = reconstructedImgs_validate[bestMatchIdx]
            avgLoss_array[idx] = best_l2
            print 'idxs %d to %d, best l2: %g from val run: %d' %(leftIdx, rightIdx, best_l2, bestMatchIdx)

            # checkpoint
            bestNoiseVecs[leftIdx:rightIdx, :] = z_hats_validate[bestMatchIdx]
            if (isTrain):
                np.save('tmp/bestNoiseVecs_%d_%d_mnist_train.npy' %(squareSideLength, rightIdx), bestNoiseVecs)
                np.save('tmp/hiddenImgs_%d_mnist_train.npy' %squareSideLength, hiddenImgs)
                np.save('tmp/generatedImgs_%d_mnist_train.npy' %squareSideLength, generatedImgs)
                np.save('tmp/reconstructedImgs_%d_mnist_train.npy' %squareSideLength, reconstructedImgs)
                np.save('tmp/L2_loss_bs%d_%d_mnist_train.npy' %(batch_size, squareSideLength), avgLoss_array)
            else:
                np.save('tmp/bestNoiseVecs_%d_%d_mnist_test.npy' %(squareSideLength, rightIdx), bestNoiseVecs)
                np.save('tmp/hiddenImgs_%d_mnist_test.npy' %squareSideLength, hiddenImgs)
                np.save('tmp/generatedImgs_%d_mnist_test.npy' %squareSideLength, generatedImgs)
                np.save('tmp/reconstructedImgs_%d_mnist_test.npy' %squareSideLength, reconstructedImgs)
                np.save('tmp/L2_loss_bs%d_%d_mnist_test.npy' %(batch_size, squareSideLength), avgLoss_array)
                print 'Saved a crap load of stuff'

        avgLoss = np.mean(avgLoss_array)
        print 'Final avg loss: %g' %avgLoss

        if (isTrain):
            np.save('mnist_gan/hiddenImgs_%d_mnist_train.npy' %squareSideLength, hiddenImgs)
            np.save('mnist_gan/generatedImgs_%d_mnist_train.npy' %squareSideLength, generatedImgs)
            np.save('mnist_gan/reconstructedImgs_%d_mnist_train.npy' %squareSideLength, reconstructedImgs)
            np.save('mnist_gan/L2_loss_bs%d_%d_mnist_train.npy' %(batch_size, squareSideLength), avgLoss_array)
            print 'Train images saved'
        else:
            np.save('mnist_gan/hiddenImgs_%d_mnist_test.npy' %squareSideLength, hiddenImgs)
            np.save('mnist_gan/generatedImgs_%d_mnist_test.npy' %squareSideLength, generatedImgs)
            np.save('mnist_gan/reconstructedImgs_%d_mnist_test.npy' %squareSideLength, reconstructedImgs)
            np.save('mnist_gan/L2_loss_bs%d_%d_mnist_test.npy' %(batch_size, squareSideLength), avgLoss_array)
            print 'Test images saved'


operation = 2
if (operation == 0):
    train()
elif (operation == 1):
    predict()
elif (operation == 2):
    image_completion()