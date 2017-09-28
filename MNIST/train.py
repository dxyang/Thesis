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

def train(maskVecXoneYzero, squareSideLength, nCols, bottleneck, n_iterations=20000):
  train, test = mnist_preprocessing.returnData()
  # train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
  # test_hideRight, Xtest_hideRight, Ytest_hideRight = mnist_preprocessing.returnHalfData(ncols=nCols)

  #train_hideLeft, Xtrain_hideLeft, Ytrain_hideLeft, \
  #test_hideLeft, Xtest_hideLeft, Ytest_hideLeft = mnist_preprocessing.returnHalfData_HideLeft(ncols=nCols)

  train_hideMiddle, Xtrain_hideMiddle, Ytrain_hideMiddle, \
  test_hideMiddle, Xtest_hideMiddle, Ytest_hideMiddle = mnist_preprocessing.returnSquareData(squareSideLength=squareSideLength)

  idxs = np.arange(55000)

  train_input = train_hideMiddle
  train_output = train

  test_input = test_hideMiddle
  test_output = test

  #Y_train = Ytrain_hideRight
  #Y_test = Ytest_hideRight 

  tf.reset_default_graph()
  net = mnist_tf.create_network_autoencoder(maskVecXoneYzero=maskVecXoneYzero, bottleneck=bottleneck)

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
      net.saver.restore(sess, os.getcwd() + "/tmp/model_bottleneck_%d.ckpt" %bottleneck)
      print("Model restored.")

      # Don't go through training
      n_iterations = 0      

    for i in range(n_iterations):
      # Shuffle the training data every time we run through the set
      if i%(55000/50) == 0:
        np.random.shuffle(idxs)
        bufferedIdxs = np.concatenate((idxs, idxs[0:50]))

      # Shift through the training set 50 items at a time
      leftIdx = (i*50)%55000
      rightIdx = leftIdx + 50
      batch_in = train_input[:, bufferedIdxs[leftIdx:rightIdx]].T
      batch_out = train_output[:, bufferedIdxs[leftIdx:rightIdx]].T

      # Train and learn :D
      sess.run(net.train_step, feed_dict={net.x: batch_in,
                                         net.y: batch_out, 
                                         net.keep_prob: 0.5})

      # Every 100 iterations print out a status
      if i%100 == 0:
        train_cost = sess.run(net.cost, feed_dict={net.x: batch_in,
                                                   net.y: batch_out, 
                                                   net.keep_prob: 1.0})
        print("step %d, l2 %g" %(i, train_cost))


      # Check if we're learning by 5000 iterations
      if (i==10000):
        train_cost = sess.run(net.cost, feed_dict={net.x: batch_in,
                                                   net.y: batch_out, 
                                                   net.keep_prob: 1.0})
        if (train_cost > 0.11):
          print("Not training :(")
          keepTraining = True
          break


    # Test and Training Accuracies
    testing_cost = sess.run(net.cost, feed_dict={net.x: test_input.T,
                                              net.y: test_output.T, 
                                              net.keep_prob: 1.0})

    training_cost_temp = np.zeros(5)
    for i in range(5):
      training_cost_temp[i] = sess.run(net.cost, feed_dict={net.x:train_input[:, i*11000:(i+1)*11000].T, 
                                                            net.y:train_output[:, i*11000:(i+1)*11000].T,
                                                            net.keep_prob: 1.0})
    training_cost = np.mean(training_cost_temp)
    print("Test Cost %g" %testing_cost)
    print("Training Cost %g" %training_cost)

    # If this was a succesful train...
    if (not keepTraining):
      # Save the model
      #modelStr = "/tmp/model_ncols_%d.ckpt" %nCols
      modelStr = "/tmp/model_square_%d.ckpt" %squareSideLength
      #modelStr = "/tmp/model_square_%d_adversarial.ckpt" %squareSideLength
      #modelStr = "/tmp/model_bottleneck_%d.ckpt" %bottleneck
      save_path = net.saver.save(sess, os.getcwd() + modelStr)
      print("Model saved in file: %s" % save_path)

      # Get splits of test set
      testSplit_in = np.zeros((10, test_input.shape[0], test_input.shape[1]/10))     # input to the conv net (hidden image)
      testSplit_out = np.zeros((10, test_output.shape[0], test_output.shape[1]/10))  # ground truth of conv net output
      for i in range(10):
        for j in range(10):
          testSplit_in[i, :, j*100:(j+1)*100] = test_input[:, j*1000+i*100:j*1000+(i+1)*100]
          testSplit_out[i, :, j*100:(j+1)*100] = test_output[:, j*1000+i*100:j*1000+(i+1)*100]

        predicted_testSplit = sess.run(net.y_conv, feed_dict={net.x:testSplit_in[i].T, 
                                                            net.y:testSplit_out[i].T,
                                                            net.keep_prob: 1.0})

        #testSplit_hat_hidden, testSplit_hat_X, testSplit_hat_Y = mnist_preprocessing.hideData(predicted_testSplit.T, maskVecXoneYzero)
        #testSplit_hidden, testSplit_X, testSplit_Y = mnist_preprocessing.hideData(testSplit_out[i], maskVecXoneYzero)

        testSplit_hat_Y = predicted_testSplit
        testSplit_Y = testSplit_out[i].T
        diff_testSplit = testSplit_hat_Y - testSplit_Y
        mses_testSplits[i] = np.mean(np.multiply(diff_testSplit, diff_testSplit))

      '''
      # Get the whole predicted test and train image matrices
      predicted_test = sess.run(net.y_conv, feed_dict={net.x:test_input.T, net.y:test_output.T, net.keep_prob: 1.0})
      predicted_train = sess.run(net.y_conv, feed_dict={net.x:train_input.T, net.y:train_output.T, net.keep_prob: 1.0})

      # Apply mask to separate out X and Y components of the image
      predictedTest_hidden, Xtest_hat_hidden, Ytest_hat_hidden = mnist_preprocessing.hideData(predicted_test.T, maskVecXoneYzero)
      predictedTrain_hidden, Xtrain_hat_hidden, Ytrain_hat_hidden = mnist_preprocessing.hideData(predicted_train.T, maskVecXoneYzero)
      
      # Calculate the MSE
      diff_test = Ytest_hat_hidden - Y_test
      mse_test_generated = np.mean(np.multiply(diff_test, diff_test))
      
      diff_train = Ytrain_hat_hidden - Y_train
      mse_train_generated = np.mean(np.multiply(diff_train, diff_train))
      '''

    return keepTraining, testing_cost, training_cost, mses_testSplits

def train_adversarial(squareSideLength, nCols, n_iterations=20000):
  train, test = mnist_preprocessing.returnData()

  train_hideMiddle, Xtrain_hideMiddle, Ytrain_hideMiddle, \
  test_hideMiddle, Xtest_hideMiddle, Ytest_hideMiddle = mnist_preprocessing.returnSquareData(squareSideLength=squareSideLength)

  idxs = np.arange(55000)

  train_input = train_hideMiddle
  train_output = Ytrain_hideMiddle

  test_input = test_hideMiddle
  test_output = Ytest_hideMiddle

  tf.reset_default_graph()
  net = mnist_tf.create_network_autoencoder_adversarial()

  # initialize things to return
  keepTraining = False
  testing_cost = -1.0
  training_cost = -1.0
  mses_testSplits = np.zeros(10)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    if (False):
      # Restore variables from disk.
      net.saver.restore(sess, os.getcwd() + "/tmp/model_bottleneck_%d.ckpt" %bottleneck)
      print("Model restored.")

      # Don't go through training
      n_iterations = 0      

    for i in range(n_iterations):
      # Shuffle the training data every time we run through the set
      if i%(55000/50) == 0:
        np.random.shuffle(idxs)
        bufferedIdxs = np.concatenate((idxs, idxs[0:50]))

      # Shift through the training set 50 items at a time
      leftIdx = (i*50)%55000
      rightIdx = leftIdx + 50
      batch_in = train_input[:, bufferedIdxs[leftIdx:rightIdx]].T
      batch_out = train_output[:, bufferedIdxs[leftIdx:rightIdx]].T

      # Train and learn :D
      switch = 0
      if (i < switch):
        sess.run(net.pretrain_step, feed_dict={net.x: batch_in,
                                           net.y: batch_out, 
                                           net.keep_prob: 0.5})
      else:
        sess.run(net.g_step, feed_dict={net.x: batch_in,
                                            net.y: batch_out, 
                                            net.keep_prob: 0.5})
        sess.run(net.d_step, feed_dict={net.x: batch_in,
                                            net.y: batch_out, 
                                            net.keep_prob: 0.5})

      # Every 100 iterations print out a status
      if i%100 == 0:
        if (i < switch):
          train_cost = sess.run(net.cost, feed_dict={net.x: batch_in,
                                                     net.y: batch_out, 
                                                     net.keep_prob: 1.0})
          print("step %d, l2 %g" %(i, train_cost))
        else:
          train_cost, train_d_loss, train_g_loss, disc_fake, disc_real = sess.run([net.cost, net.d_loss, net.g_loss, net.avg_disc_fake, net.avg_disc_real], feed_dict={net.x: batch_in,
                                                       net.y: batch_out, 
                                                       net.keep_prob: 1.0})
          print("step %d, l2 %g, adv_d %g, adv_g %g" %(i, train_cost, train_d_loss, train_g_loss))
          print("*****disc_fake %g, disc_real %g" %(disc_fake, disc_real))
  

      # Check if we're learning by 10000 iterations
      if (i==10000):
        train_cost = sess.run(net.cost, feed_dict={net.x: batch_in,
                                                   net.y: batch_out, 
                                                   net.keep_prob: 1.0})
        if (train_cost > 0.09):
          print("Not training :(")
          keepTraining = True
          break

    # Test and Training Accuracies
    testing_cost = sess.run(net.cost, feed_dict={net.x: test_input.T,
                                              net.y: test_output.T, 
                                              net.keep_prob: 1.0})

    training_cost_temp = np.zeros(5)
    for i in range(5):
      training_cost_temp[i] = sess.run(net.cost, feed_dict={net.x:train_input[:, i*11000:(i+1)*11000].T, 
                                                            net.y:train_output[:, i*11000:(i+1)*11000].T,
                                                            net.keep_prob: 1.0})
    training_cost = np.mean(training_cost_temp)
    print("Test Cost %g" %testing_cost)
    print("Training Cost %g" %training_cost)

    # If this was a succesful train...
    if (not keepTraining):
      # Save the model
      modelStr = "/tmp/model_square_%d_adversarial.ckpt" %squareSideLength
      save_path = net.saver.save(sess, os.getcwd() + modelStr)
      print("Model saved in file: %s" % save_path)

      # Get splits of test set
      testSplit_in = np.zeros((10, test_input.shape[0], test_input.shape[1]/10))     # input to the conv net (hidden image)
      testSplit_out = np.zeros((10, test_output.shape[0], test_output.shape[1]/10))  # ground truth of conv net output
      for i in range(10):
        for j in range(10):
          testSplit_in[i, :, j*100:(j+1)*100] = test_input[:, j*1000+i*100:j*1000+(i+1)*100]
          testSplit_out[i, :, j*100:(j+1)*100] = test_output[:, j*1000+i*100:j*1000+(i+1)*100]

        predicted_testSplit = sess.run(net.y_conv, feed_dict={net.x:testSplit_in[i].T, 
                                                            net.y:testSplit_out[i].T,
                                                            net.keep_prob: 1.0})

        testSplit_hat_Y = predicted_testSplit
        testSplit_Y = testSplit_out[i].T
        diff_testSplit = testSplit_hat_Y - testSplit_Y
        mses_testSplits[i] = np.mean(np.multiply(diff_testSplit, diff_testSplit))

    return keepTraining, testing_cost, training_cost, mses_testSplits


numTrials = 1

test_costs_data = np.zeros((numTrials))
train_costs_data = np.zeros((numTrials))
#mses_test_generated_data = np.zeros((numTrials))
#mses_train_generated_data = np.zeros((numTrials))
mses_testSplits_data = np.zeros((numTrials, 10))

# all columns: np.arange(1, 29)
# all squares: np.arange(4, 30, 2)
for i in range(numTrials):
  keepTraining = True

  while (keepTraining):
    nCols = 0
    squareSideLength = 20
    bottleneck = 128
    maskVec = mnist_preprocessing.generateCenterSquareMask(squareSideLength) #mnist_preprocessing.generateColumnMask(nCols) #mnist_preprocessing.generateColumnMask_FromLeft(nCols)
    #print '------------Square Size: %d x %d -------------' %(squareSideLength , squareSideLength)
    #print '------------Columns Removed: %d -------------' %nCols
    #print '------------Bottleneck: %d -------------' %bottleneck

    #mses_test_generated_data[i], mses_train_generated_data[i], \
    keepTraining, test_costs_data[i], train_costs_data[i], \
    mses_testSplits_data[i] = train(maskVecXoneYzero= maskVec, squareSideLength=squareSideLength, nCols = nCols, bottleneck=bottleneck)

    # keepTraining, test_costs_data[i], train_costs_data[i], \
    # mses_testSplits_data[i] = train_adversarial(squareSideLength=squareSideLength, nCols = nCols)

  print '----- Test and Train Costs (MSE of generated image, whole dataset ----'
  print test_costs_data
  print train_costs_data
  print '----- Test and Train Costs (MSE of generated image, whole dataset ----'
  #print mses_test_generated_data
  #print mses_train_generated_data
  print '----- Test Splits (MSE of generated image, test splits ----'
  print mses_testSplits_data

#np.save('mnist_data_test_costs_ncols_bottleneck.npy', test_costs_data)
#np.save('mnist_data_train_costs_ncols_bottleneck.npy', train_costs_data)
#np.save('mnist_data_mses_test_generated_LtoR.npy', mses_test_generated_data)
#np.save('mnist_data_mses_train_generated_LtoR.npy', mses_train_generated_data)
#np.save('mnist_data_mses_testSplits_ncols_bottleneck.npy', mses_testSplits_data)
