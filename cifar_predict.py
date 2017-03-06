import tensorflow as tf
import numpy as np
import os
import cifar_tf
import cifar

def predict(ncols, squareSideLength):
  train, test = cifar.returnCIFARdata()
  train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
  test_hideRight, Xtest_hideRight, Ytest_hideRight = cifar.returnHalfData(ncols=ncols)

  X_input = train_hideRight
  Y_output = train

  test_X_input = test_hideRight
  test_Y_output = test

  maskVecXoneYzero = cifar.generateColumnMask(ncols)
  tf.reset_default_graph()
  net = cifar_tf.create_autoencoder(maskVecXoneYzero=maskVecXoneYzero, bottleneck=384)
  with tf.Session() as sess:
    # Restore variables from disk.
    net.saver.restore(sess, os.getcwd() + "/tmp/model_cifar_ncols_%d.ckpt" %ncols)
    print("Model restored.")

    # Whole Test and Training Accuracies
    testing_cost = sess.run(net.cost, feed_dict={net.x: test_X_input.T,
                                              net.y: test_Y_output.T, 
                                              net.keep_prob: 1.0})
    training_cost_temp = np.zeros(5)
    for i in range(5):
      training_cost_temp[i] = sess.run(net.cost, feed_dict={net.x:X_input[:, i*10000:(i+1)*10000].T, 
                                                            net.y:Y_output[:, i*10000:(i+1)*10000].T,
                                                            net.keep_prob: 1.0})
    training_cost = np.mean(training_cost_temp)

    print("Final Test Cost %g" %testing_cost)
    print("Final Training Cost %g" %training_cost)

    # Generate stuff - hidden data
    predicted_test = sess.run(net.y_conv, feed_dict={net.x:test_X_input.T, net.y:test_Y_output.T, net.keep_prob: 1.0})
    predicted_train = np.zeros((50000, 3072))
    for i in range(5):
      predicted_train[i*10000:(i+1)*10000, :] = sess.run(net.y_conv, feed_dict={net.x:X_input[:, i*10000:(i+1)*10000].T, 
                                                      net.y:Y_output[:, i*10000:(i+1)*10000].T, 
                                                      net.keep_prob: 1.0})

    np.save('predictedTest_cifar_ncols_%d.npy' %ncols, predicted_test.T)
    np.save('predictedTrain_cifar_ncols_%d.npy' %ncols, predicted_train.T)

predict(ncols=16, squareSideLength=0)