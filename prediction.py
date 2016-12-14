import tensorflow as tf
import numpy as np
import os
import mnist_preprocessing
import mnist_tf

def predict_original():
  train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
  test_hideRight, Xtest_hideRight, Ytest_hideRight = mnist_preprocessing.returnHalfData(endBuffer=False)

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
  train, test = mnist_preprocessing.returnData(endBuffer=False)

  train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
  test_hideRight, Xtest_hideRight, Ytest_hideRight = mnist_preprocessing.returnHalfData(endBuffer=False)

  net = mnist_tf.create_network_autoencoder()
  with tf.Session() as sess:
    # Restore variables from disk.
    net.saver.restore(sess, os.getcwd() + "/tmp/model.ckpt")
    print("Model restored.")

    # Test and Training Accuracies
    test_cost = sess.run(net.cost, feed_dict={net.x:test.T, net.keep_prob: 1.0})
    print("Final Test MSE %g" %test_cost)
    training_cost = sess.run(net.cost, feed_dict={net.x:train.T, net.keep_prob: 1.0})
    print("Final Training MSE %g" %training_cost)

    # Generate stuff - make sure the autoencoder works 
    predicted_train_regeneration = sess.run(net.y_conv, feed_dict={net.x:train.T, net.keep_prob: 1.0})
    predicted_test_regeneration = sess.run(net.y_conv, feed_dict={net.x:test.T, net.keep_prob: 1.0})

    np.save('predictedTrain_regeneration.npy', predicted_train_regeneration.T)
    np.save('predictedTest_regeneration.npy', predicted_test_regeneration.T)

    # Generate stuff - hidden data
    predicted_train = sess.run(net.y_conv, feed_dict={net.x:train_hideRight.T, net.keep_prob: 1.0})
    predicted_test = sess.run(net.y_conv, feed_dict={net.x:test_hideRight.T, net.keep_prob: 1.0})

    np.save('predictedTrain.npy', predicted_train.T)
    np.save('predictedTest.npy', predicted_test.T)

predict()