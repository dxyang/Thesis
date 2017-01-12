import tensorflow as tf
import numpy as np
import os
import mnist_tf
import mnist_preprocessing

def train_original(n_iterations=1000):
  train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
  test_hideRight, Xtest_hideRight, Ytest_hideRight = mnist_preprocessing.returnHalfData()

  #net = mnist_tf.create_network_basic()
  #net = mnist_tf.create_network_batchnorm()
  #net = mnist_tf.create_network_fullyconnected()
  net = mnist_tf.create_network_kinda_autoencoder()
  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter(
                        os.getcwd() + "/tb_out/", graph=sess.graph)

    sess.run(tf.initialize_all_variables())

    for i in range(n_iterations):
      leftIdx = (i*50)%10000
      rightIdx = leftIdx + 50

      batch_X = Xtrain_hideRight[:, leftIdx:rightIdx].T
      batch_Y = Ytrain_hideRight[:, leftIdx:rightIdx].T

      sess.run(net.train_step, feed_dict={net.x: batch_X, 
                                          net.y_: batch_Y, 
                                          net.keep_prob: 0.5})

      if i%100 == 0:
        '''
        summary, image_summary = sess.run([net.summary_op, net.image_summary_op], 
                           feed_dict={net.x: batch_X, 
                                      net.y_: batch_Y, 
                                      net.keep_prob: 1.0})
        #image_summary = sess.run(net.image_summary_op, 
        #                         feed_dict={net.x: batch_X, 
        #                                    net.y_: batch_Y, 
        #                                    net.keep_prob: 1.0})
        
        summary_writer.add_summary(summary, i)
        summary_writer.add_summary(image_summary, i)
        '''
        train_mse = sess.run(net.mse, feed_dict={net.x: batch_X, 
                                                 net.y_: batch_Y, 
                                                 net.keep_prob: 1.0})
        print("step %d, batch avg mse %g"%(i, train_mse))

    # Test and Training Accuracies
    test_mse = sess.run(net.mse, feed_dict={net.x:Xtest_hideRight.T, 
                                            net.y_:Ytest_hideRight.T, 
                                            net.keep_prob: 1.0})

    training_mse = sess.run(net.mse, feed_dict={net.x:Xtrain_hideRight[:, :10000].T, 
                                                net.y_:Ytrain_hideRight[:, :10000].T, 
                                                net.keep_prob: 1.0})

    print("Final Test MSE %g" %test_mse)
    print("Final Training MSE %g" %training_mse)

    # Save the variables to disk.
    save_path = net.saver.save(sess, os.getcwd() + "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

def train(squareSideLength, ncols, bottleneck, n_iterations=20000):
  train, test = mnist_preprocessing.returnData()
  train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
  test_hideRight, Xtest_hideRight, Ytest_hideRight = mnist_preprocessing.returnHalfData(ncols=ncols)

  #train_hideMiddle, Xtrain_hideMiddle, Ytrain_hideMiddle, \
  #test_hideMiddle, Xtest_hideMiddle, Ytest_hideMiddle = mnist_preprocessing.returnSquareData(squareSideLength=squareSideLength)

  idxs = np.arange(10000)

  X_input = train_hideRight
  Y_output = train

  test_X_input = test_hideRight
  test_Y_output = test

  tf.reset_default_graph()
  net = mnist_tf.create_network_autoencoder(bottleneck=bottleneck)

  # initialize things to return
  keepTraining = False
  testing_cost = -1.0
  training_cost = -1.0
  mse_test_generated = -1.0
  mse_train_generated = -1.0
  mses_testSplits = np.zeros(10)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(n_iterations):
      # Shuffle the training data every time we run through the set
      if i%200 == 0:
        np.random.shuffle(idxs)
        bufferedIdxs = np.concatenate((idxs, idxs[0:50]))

      # Shift through the training set 50 items at a time
      leftIdx = (i*50)%10000
      rightIdx = leftIdx + 50
      batch_X = X_input[:, bufferedIdxs[leftIdx:rightIdx]].T
      batch_Y = Y_output[:, bufferedIdxs[leftIdx:rightIdx]].T

      # Train and learn :D
      sess.run(net.train_step, feed_dict={net.x: batch_X,
                                          net.y: batch_Y, 
                                          net.keep_prob: 0.5})

      # Every 100 iterations print out a status
      if i%100 == 0:
        train_cost = sess.run(net.cost, feed_dict={net.x: batch_X,
                                                   net.y: batch_Y, 
                                                   net.keep_prob: 1.0})
        print("step %d, batch avg l2 %g"%(i, train_cost))


      # Check if we're learning by 5000 iterations
      if (i==5000):
        train_cost = sess.run(net.cost, feed_dict={net.x: batch_X,
                                                   net.y: batch_Y, 
                                                   net.keep_prob: 1.0})
      
        if (train_cost > 0.05):
          print("Not training :(")
          keepTraining = True
          break


    # Test and Training Accuracies
    testing_cost = sess.run(net.cost, feed_dict={net.x: test_X_input.T,
                                              net.y: test_Y_output.T, 
                                              net.keep_prob: 1.0})

    training_cost = sess.run(net.cost, feed_dict={net.x:X_input.T, 
                                                  net.y:Y_output.T,
                                                  net.keep_prob: 1.0})
    print("Final Whole Image Test Cost %g" %testing_cost)
    print("Final Whole Image Training Cost %g" %training_cost)

    # If this was a succesful train...
    if (not keepTraining):
      # Save the model
      modelStr = "/tmp/model.ckpt"
      save_path = net.saver.save(sess, os.getcwd() + modelStr)
      print("Model saved in file: %s" % save_path)

      # Get splits of test set
      testSplit_in = np.zeros((10, test_X_input.shape[0], test_X_input.shape[1]/10))     # input to the conv net (hidden image)
      testSplit_out = np.zeros((10, test_Y_output.shape[0], test_Y_output.shape[1]/10))  # ground truth of conv net output
      for i in range(10):
        for j in range(10):
          testSplit_in[i, :, j*10:(j+1)*10] = test_X_input[:, j*100+i*10:j*100+(i+1)*10]
          testSplit_out[i, :, j*10:(j+1)*10] = test_Y_output[:, j*100+i*10:j*100+(i+1)*10]

        predicted_testSplit = sess.run(net.y_conv, feed_dict={net.x:testSplit_in[i].T, 
                                                            net.y:testSplit_out[i].T,
                                                            net.keep_prob: 1.0})

        testSplit_hat_hidden, testSplit_hat_X, testSplit_hat_Y = mnist_preprocessing.hideData(
          predicted_testSplit.T, mnist_preprocessing.generateColumnMask(ncols)) #mnist_preprocessing.generateCenterSquareMask(squareSideLength))
        testSplit_hidden, testSplit_X, testSplit_Y = mnist_preprocessing.hideData(
          testSplit_out[i], mnist_preprocessing.generateColumnMask(ncols)) #mnist_preprocessing.generateCenterSquareMask(squareSideLength))

        diff_testSplit = testSplit_hat_Y - testSplit_Y
        mses_testSplits[i] = np.mean(np.multiply(diff_testSplit, diff_testSplit))

      # Get the whole predicted test and train image matrices
      predicted_test = sess.run(net.y_conv, feed_dict={net.x:test_X_input.T, net.y:test_Y_output.T, net.keep_prob: 1.0})
      predicted_train = sess.run(net.y_conv, feed_dict={net.x:X_input.T, net.y:Y_output.T, net.keep_prob: 1.0})

      # Apply mask to separate out X and Y components of the image
      predictedTest_hidden, Xtest_hat_hidden, Ytest_hat_hidden = mnist_preprocessing.hideData(
        predicted_test.T, mnist_preprocessing.generateColumnMask(ncols)) #mnist_preprocessing.generateCenterSquareMask(squareSideLength))
      predictedTrain_hidden, Xtrain_hat_hidden, Ytrain_hat_hidden = mnist_preprocessing.hideData(
        predicted_train.T, mnist_preprocessing.generateColumnMask(ncols)) #mnist_preprocessing.generateCenterSquareMask(squareSideLength))
      
      # Calculate the MSE
      diff_test = Ytest_hat_hidden - Ytest_hideRight
      mse_test_generated = np.mean(np.multiply(diff_test, diff_test))
      
      diff_train = Ytrain_hat_hidden - Ytrain_hideRight
      mse_train_generated = np.mean(np.multiply(diff_train, diff_train))

    return keepTraining, testing_cost, training_cost, mse_test_generated, mse_train_generated, mses_testSplits

numTrials = 1

test_costs_data = np.zeros((numTrials))
train_costs_data = np.zeros((numTrials))
mses_test_generated_data = np.zeros((numTrials))
mses_train_generated_data = np.zeros((numTrials))
mses_testSplits_data = np.zeros((numTrials, 10))

for i in range(numTrials):
  keepTraining = True

  while (keepTraining):
    #print '------------Square Size: %d x %d -------------' %(squareSideLength , squareSideLength)
    #print '------------Columns Removed: %d -------------' %nCols
    #print '------------Bottleneck: %d -------------' %bottleneck

    keepTraining, test_costs_data[i], train_costs_data[i], \
    mses_test_generated_data[i], mses_train_generated_data[i], \
    mses_testSplits_data[i] = train(squareSideLength=0, ncols = 14, bottleneck=128)

  print '----- Test and Train Costs (MSE of whole image, whole dataset ----'
  print test_costs_data
  print train_costs_data
  print '----- Test and Train Costs (MSE of generated image, whole dataset ----'
  print mses_test_generated_data
  print mses_train_generated_data
  print '----- Test Splits (MSE of generated image, test splits ----'
  print mses_testSplits_data

'''
np.save('data_test_costs_LtoR.npy', test_costs_data)
np.save('data_train_costs_LtoR.npy', train_costs_data)
np.save('data_mses_test_generated_LtoR.npy', mses_test_generated_data)
np.save('data_mses_train_generated_LtoR.npy', mses_train_generated_data)
np.save('data_mses_testSplits_LtoR.npy', mses_testSplits_data)
'''