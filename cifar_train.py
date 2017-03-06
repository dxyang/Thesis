import tensorflow as tf
import numpy as np
import os
import cifar_tf
import cifar

def train(maskVecXoneYzero, squareSideLength, nCols, bottleneck, n_iterations=20000):
  train, test, train_labels, test_labels = cifar.returnCIFARdata()
  train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
  test_hideRight, Xtest_hideRight, Ytest_hideRight = cifar.returnHalfData(ncols=nCols)

  idxs = np.arange(50000)

  train_input = train_hideRight
  train_output = train

  test_input = test_hideRight
  test_output = test

  Y_test = Ytrain_hideRight 
  Y_train = Ytest_hideRight

  tf.reset_default_graph()
  net = cifar_tf.create_autoencoder(maskVecXoneYzero=maskVecXoneYzero, bottleneck=bottleneck)

  # initialize things to return
  keepTraining = False
  testing_cost = -1.0
  training_cost = -1.0
  #mse_test_generated = -1.0
  #mse_train_generated = -1.0
  mses_testSplits = np.zeros(10)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    if (False):
      # Restore variables from disk.
      net.saver.restore(sess, os.getcwd() + "/tmp/model_cifar_ncols_%d.ckpt" %nCols)
      print("Model restored.")

      # Don't go through training
      n_iterations = 0      

    for i in range(n_iterations):
      # Shuffle the training data every time we run through the set
      if i%(50000/50) == 0:
        np.random.shuffle(idxs)
        bufferedIdxs = np.concatenate((idxs, idxs[0:50]))

      # Shift through the training set 50 items at a time
      leftIdx = (i*50)%50000
      rightIdx = leftIdx + 50
      batch_in = train_input[:, bufferedIdxs[leftIdx:rightIdx]].T
      batch_out = train_output[:, bufferedIdxs[leftIdx:rightIdx]].T
      batch_labels = train_labels[bufferedIdxs[leftIdx:rightIdx]]

      # Train and learn :D
      sess.run(net.train_step, feed_dict={net.x: batch_in,
                                          net.y: batch_out,
                                          net.labels: batch_labels, 
                                          net.keep_prob: 0.5})

      # Every 100 iterations print out a status
      if i%100 == 0:
        train_cost, accuracy = sess.run([net.cost, net.accuracy], feed_dict={net.x: batch_in,
                                                   net.y: batch_out, 
                                                   net.labels: batch_labels, 
                                                   net.keep_prob: 1.0})
        print("step %d, batch avg l2 %g, accuracy %g"%(i, train_cost, accuracy))


      # Check if we're learning by 10000 iterations
      if (i==20000):
        train_cost = sess.run(net.cost, feed_dict={net.x: batch_in,
                                                   net.y: batch_out, 
                                                   net.labels: batch_labels, 
                                                   net.keep_prob: 1.0})
        if (train_cost > 0.05):
          print("Not training :(")
          keepTraining = True
          break

    # Test and Training Accuracies
    testing_cost_temp = np.zeros(2)
    testing_acc_temp = np.zeros(2)
    for i in range(2):
      testing_cost_temp[i], testing_acc_temp[i] = sess.run([net.cost, net.accuracy], feed_dict={net.x: test_input[:, i*5000:(i+1)*5000].T,
                                                           net.y: test_output[:, i*5000:(i+1)*5000].T,
                                                           net.labels: test_labels[i*5000:(i+1)*5000], 
                                                           net.keep_prob: 1.0})
    testing_cost = np.mean(testing_cost_temp)
    testing_acc = np.mean(testing_acc_temp)

    training_cost_temp = np.zeros(10)
    training_acc_temp = np.zeros(10)
    for i in range(10):
      training_cost_temp[i], training_acc_temp[i] = sess.run([net.cost, net.accuracy], feed_dict={net.x:train_input[:, i*5000:(i+1)*5000].T, 
                                                            net.y:train_output[:, i*5000:(i+1)*5000].T,
                                                            net.labels: train_labels[i*5000:(i+1)*5000], 
                                                            net.keep_prob: 1.0})
    training_cost = np.mean(training_cost_temp)
    training_acc = np.mean(training_acc_temp)
    print("Predicted Image Portion Test Cost %g" %testing_cost)
    print("Predicted Image Portion Test Acc %g" %testing_acc)
    print("Predicted Image Portion Training Cost %g" %training_cost)
    print("Predicted Image Portion Training Acc %g" %training_acc)

    # If this was a succesful train...
    if (not keepTraining):
      # Save the model
      modelStr = "/tmp/model_cifar_ncols_%d.ckpt" %nCols
      save_path = net.saver.save(sess, os.getcwd() + modelStr)
      print("Model saved in file: %s" % save_path)

      # Get splits of test set
      testSplit_in = np.zeros((10, test_input.shape[0], test_input.shape[1]/10))     # input to the conv net (hidden image)
      testSplit_out = np.zeros((10, test_output.shape[0], test_output.shape[1]/10))  # ground truth of conv net output
      for i in range(10):
        testSplit_in[i, :, :] = test_input[:, i*1000:(i+1)*1000]
        testSplit_out[i, :, :] = test_output[:, i*1000:(i+1)*1000]

        predicted_testSplit, acc = sess.run([net.cost, net.accuracy], feed_dict={net.x:testSplit_in[i].T, 
                                                            net.y:testSplit_out[i].T,
                                                            net.labels: test_labels[i*1000:(i+1)*1000],
                                                            net.keep_prob: 1.0})
        print acc
        testSplit_hat_hidden, testSplit_hat_X, testSplit_hat_Y = cifar.hideData(predicted_testSplit.T, maskVecXoneYzero)
        testSplit_hidden, testSplit_X, testSplit_Y = cifar.hideData(testSplit_out[i], maskVecXoneYzero)

        diff_testSplit = testSplit_hat_Y - testSplit_Y
        mses_testSplits[i] = np.mean(np.multiply(diff_testSplit, diff_testSplit))

    return keepTraining, testing_cost, training_cost, mses_testSplits





keepTraining = True
while (keepTraining):
  nCols = 16
  vecMask = cifar.generateColumnMask(nCols)
  keepTraining, temp1, temp2, temp3 = train(maskVecXoneYzero = vecMask, squareSideLength = 0, nCols = nCols, bottleneck = 512)
