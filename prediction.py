import tensorflow as tf
import numpy as np
import os
import mnist_preprocessing
import mnist_tf

train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
test_hideRight, Xtest_hideRight, Ytest_hideRight = mnist_preprocessing.returnData()

def predict():
  net = mnist_tf.create_network()
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # Restore variables from disk.
    net.saver.restore(sess, os.getcwd() + "/tmp/model.ckpt")
    print("Model restored.")

    predicted_train = sess.run(net.y_conv, feed_dict={net.x:train_hideRight.T, net.y_:Ytrain_hideRight.T, net.keep_prob: 1.0})
    predicted_test = sess.run(net.y_conv, feed_dict={net.x:test_hideRight.T, net.y_:Ytest_hideRight.T, net.keep_prob: 1.0})

    # Test and Training Accuracies
    test_mse = sess.run(net.mse, feed_dict={net.x:test_hideRight.T, net.y_:Ytest_hideRight.T, net.keep_prob: 1.0})
    print("Final Test MSE %g" %test_mse)
    training_mse = sess.run(net.mse, feed_dict={net.x:train_hideRight.T, net.y_:Ytrain_hideRight.T, net.keep_prob: 1.0})
    print("Final Training MSE %g" %training_mse)

    # Save predicted data
    np.save('predictedTrain_6.npy', predicted_train)
    np.save('predictedTest_6.npy', predicted_test)

predict()