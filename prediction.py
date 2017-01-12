import tensorflow as tf
import numpy as np
import os
import mnist_preprocessing
import mnist_tf

def predict_original():
  train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
  test_hideRight, Xtest_hideRight, Ytest_hideRight = mnist_preprocessing.returnHalfData()

  #net = mnist_tf.create_network_basic()
  #net = mnist_tf.create_network_batchnorm()
  #net = mnist_tf.create_network_fullyconnected()
  net = mnist_tf.create_network_kinda_autoencoder()
  with tf.Session() as sess:
    # Restore variables from disk.
    net.saver.restore(sess, os.getcwd() + "/tmp/model.ckpt")
    print("Model restored.")

    predicted_train = sess.run(net.y_conv, feed_dict={net.x:Xtrain_hideRight.T, net.y_:Ytrain_hideRight.T, net.keep_prob: 1.0})
    predicted_test = sess.run(net.y_conv, feed_dict={net.x:Xtest_hideRight.T, net.y_:Ytest_hideRight.T, net.keep_prob: 1.0})

    # Test and Training Accuracies
    test_mse = sess.run(net.mse, feed_dict={net.x:Xtest_hideRight.T, net.y_:Ytest_hideRight.T, net.keep_prob: 1.0})
    print("Final Test MSE %g" %test_mse)
    training_mse = sess.run(net.mse, feed_dict={net.x:Xtrain_hideRight.T, net.y_:Ytrain_hideRight.T, net.keep_prob: 1.0})
    print("Final Training MSE %g" %training_mse)

    np.save('predictedTrain.npy', predicted_train.T)
    np.save('predictedTest.npy', predicted_test.T)


def predict():
  train, test = mnist_preprocessing.returnData()
  #train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
  #test_hideRight, Xtest_hideRight, Ytest_hideRight = mnist_preprocessing.returnHalfData(ncols=6)

  train_hideMiddle, Xtrain_hideMiddle, Ytrain_hideMiddle, \
  test_hideMiddle, Xtest_hideMiddle, Ytest_hideMiddle = mnist_preprocessing.returnSquareData(squareSideLength=8)

  X_input = train_hideMiddle
  Y_output = train

  test_X_input = test_hideMiddle
  test_Y_output = test

  net = mnist_tf.create_network_autoencoder(bottleneck=128)
  with tf.Session() as sess:
    # Restore variables from disk.
    net.saver.restore(sess, os.getcwd() + "/tmp/model_squareLength_8.ckpt")
    print("Model restored.")

    # Whole Test and Training Accuracies
    testing_cost = sess.run(net.cost, feed_dict={net.x: test_X_input.T,
                                              net.y: test_Y_output.T, 
                                              net.keep_prob: 1.0})

    training_cost = sess.run(net.cost, feed_dict={net.x:X_input.T, 
                                                  net.y:Y_output.T,
                                                  net.keep_prob: 1.0})

    print("Final Test Cost %g" %testing_cost)
    print("Final Training Cost %g" %training_cost)

    '''
    # Generate stuff - make sure the autoencoder works 
    predicted_train_sanity = sess.run(net.y_conv, feed_dict={net.x:train.T, net.y:train.T, net.keep_prob: 1.0})
    predicted_test_sanity = sess.run(net.y_conv, feed_dict={net.x:test.T, net.y:test.T, net.keep_prob: 1.0})

    np.save('predictedTrain_sanity.npy', predicted_train_sanity.T)
    np.save('predictedTest_sanity.npy', predicted_test_sanity.T)
    '''

    # Generate stuff - hidden data
    predicted_test = sess.run(net.y_conv, feed_dict={net.x:test_X_input.T, net.y:test_Y_output.T, net.keep_prob: 1.0})
    predicted_train = sess.run(net.y_conv, feed_dict={net.x:X_input.T, net.y:Y_output.T, net.keep_prob: 1.0})

    np.save('predictedTest.npy', predicted_test.T)
    np.save('predictedTrain.npy', predicted_train.T)

predict()