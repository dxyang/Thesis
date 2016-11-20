import tensorflow as tf
import numpy as np

# packing and unpacking images
def packcw(A,nr,nc):
    x = (A.T).reshape(nr*nc,1)
    return x

def unpackcw(x,nr,nc):
    A = x.reshape(nc,nr)
    return A.T

def packrw(A,nr,nc):
    x = A.reshape(nr*nc,1)
    return x

def unpackrw(x,nr,nc):
    A = x.reshape(nr,nc)
    return A

# generates a 784 element mask with columns deleted
# from the right side of the image towards the left
def generateColumnMask(delColumns):
    mask = np.ones((28, 28))
    mask[:, (28 - delColumns):] = 0
    maskVec = packcw(mask, 28, 28)
    return maskVec

# generate a 784 element mask with a square, with side
# length (sideLength), zero'd out of the middle
def generateCenterSquareMask(sideLength):
    mask = np.ones((28, 28))
    leftIdx = (28 - sideLength)/2
    rightIdx = (28 + sideLength)/2
    mask[leftIdx:rightIdx, leftIdx:rightIdx] = 0
    maskVec = packcw(mask, 28, 28)
    return maskVec

# zero out indices of a vector for a data matrix
# for a given mask
def hideData(data, mask):
    # copy the data
    newData = data.copy()
    
    # get indices from the mask
    x_idx = np.where([mask==1])[1]
    y_idx = np.where([mask==0])[1]
    
    # apply the mask
    newData[y_idx, :] = 0
    
    return newData, data[x_idx, :], data[y_idx, :]

# let's get our data
train = np.load('MNISTcwtrain1000.npy')
train = train.astype(float)/255
test = np.load('MNISTcwtest100.npy')
test = test.astype(float)/255

size = train.shape[0]
n_train = train.shape[1]
n_test = test.shape[1]

print '----MNIST dataset loaded----'
print 'Train data: %d x %d' %(size, n_train)
print 'Test data: %d x %d' %(size, n_test)

train_hideRight, Xtrain_hideRight, Ytrain_hideRight = hideData(train, generateColumnMask(14))
test_hideRight, Xtest_hideRight, Ytest_hideRight = hideData(test, generateColumnMask(14))

# x = every image with zero'd out section as training data
x = tf.placeholder(tf.float32, shape=[None, 784])

# y_ = 14 columns that are being zero'd out
y_ = tf.placeholder(tf.float32, shape=[None, 14*28])

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

# first layer of the CNN
filtersize = 7
W_conv1 = weight_variable([filtersize, filtersize, 1, 32])
b_conv1 = bias_variable([32])

# reshape the x data vector into a 28 by 28 image
x_image = tf.reshape(x, [-1,28,28,1])

# convolve x with the weight matrix and add in the bias
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# densely connected layer
W_fc1 = weight_variable([28 * 28 * 32, 1024])
b_fc1 = bias_variable([1024])

# reshape vector from convolution
h_conv1_flat = tf.reshape(h_conv1, [-1, 28 * 28 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

# read out layer
W_fc2 = weight_variable([1024, 14 * 28])
b_fc2 = bias_variable([14 * 28])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

# train
cost = tf.nn.l2_loss(y_conv - y_)
train_step = tf.train.AdamOptimizer(2e-5).minimize(cost)

mse = tf.reduce_mean(tf.mul(y_conv - y_, y_conv - y_))

# initialize variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
  sess.run(init)

  for i in range(20000):
    randomIdxs = np.random.randint(0, 10000, 50)
    batch_X = train_hideRight[:, randomIdxs].T
    batch_Y = Ytrain_hideRight[:, randomIdxs].T
    if i%100 == 0:
      train_mse = sess.run(mse, feed_dict={x: batch_X, y_: batch_Y})
      print("step %d, batch average mean square error %g"%(i, train_mse))
    sess.run(train_step, feed_dict={x: batch_X, y_: batch_Y})
  test_mse = sess.run(mse, feed_dict={x:test_hideRight.T, y_:Ytest_hideRight.T})
  print("Final Test MSE %g" %test_mse)

  predicted_train = sess.run(y_conv, feed_dict={x:train_hideRight.T, y_:Ytrain_hideRight.T})
  predicted_test = sess.run(y_conv, feed_dict={x:test_hideRight.T, y_:Ytest_hideRight.T})

  print predicted_train.shape
  print predicted_test.shape
  np.save('predictedTrain_%dx%d.npy' % (filtersize, filtersize), predicted_train)
  np.save('predictedTest_%dx%d.npy' % (filtersize, filtersize), predicted_test)