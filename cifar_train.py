import tensorflow as tf
import numpy as np
import os
import cifar_tf
import cifar

def train_classification(bottleneck, n_iterations=40000):
  train, test, train_labels, test_labels = cifar.returnCIFARdata()
  idxs = np.arange(50000)

  train_input = train
  train_output = train_labels

  test_input = test
  test_output = test_labels

  tf.reset_default_graph()
  net = cifar_tf.create_autoencoder_classification(bottleneck=bottleneck)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(n_iterations):
      # Shuffle the training data every time we run through the set
      if i%(50000/50) == 0:
        np.random.shuffle(idxs)
        bufferedIdxs = np.concatenate((idxs, idxs[0:50]))

      # Shift through the training set 50 items at a time
      leftIdx = (i*50)%50000
      rightIdx = leftIdx + 50
      batch_in = train_input[:, bufferedIdxs[leftIdx:rightIdx]].T
      batch_out = train_output[bufferedIdxs[leftIdx:rightIdx]]

      # Train and learn :D
      sess.run(net.train_step, feed_dict={net.x: batch_in,
                                          net.labels: batch_out,
                                          net.keep_prob: 0.5})

      # Every 100 iterations print out a status
      if i%100 == 0:
        cost, accuracy = sess.run([net.cost, net.accuracy], feed_dict={net.x: batch_in,
                                                   net.labels: batch_out, 
                                                   net.keep_prob: 1.0})
        print("step %d, batch classification cost %g and accuracy %g" %(i, cost, accuracy))


      if i%10000 == 0:
        # Test and Training Accuracies
        testing_cost_temp = np.zeros(2)
        testing_acc_temp = np.zeros(2)
        for j in range(2):
          testing_cost_temp[j], testing_acc_temp[j] = sess.run([net.cost, net.accuracy], 
                                                                feed_dict={net.x: test_input[:, j*5000:(j+1)*5000].T,
                                                                          net.labels: test_output[j*5000:(j+1)*5000],
                                                                          net.keep_prob: 1.0})
        testing_cost = np.mean(testing_cost_temp)
        testing_acc = np.mean(testing_acc_temp)

        training_cost_temp = np.zeros(10)
        training_acc_temp = np.zeros(10)
        for j in range(10):
          training_cost_temp[j], training_acc_temp[j] = sess.run([net.cost, net.accuracy], 
                                                                  feed_dict={net.x:train_input[:, j*5000:(j+1)*5000].T, 
                                                                             net.labels:train_output[j*5000:(j+1)*5000],
                                                                             net.keep_prob: 1.0})
        training_cost = np.mean(training_cost_temp)
        training_acc = np.mean(training_acc_temp)
        print "step %d" %i
        print("Predicted Image Portion Test Cost %g, Accuracy %g" %(testing_cost, testing_acc))
        print("Predicted Image Portion Training Cost %g, Accuracy %g" %(training_cost, training_acc))
    

    # Test and Training Accuracies
    testing_cost_temp = np.zeros(2)
    testing_acc_temp = np.zeros(2)
    for i in range(2):
      testing_cost_temp[i], testing_acc_temp[i] = sess.run([net.cost, net.accuracy], 
                                                            feed_dict={net.x: test_input[:, i*5000:(i+1)*5000].T,
                                                                      net.labels: test_output[i*5000:(i+1)*5000],
                                                                      net.keep_prob: 1.0})
    testing_cost = np.mean(testing_cost_temp)
    testing_acc = np.mean(testing_acc_temp)

    training_cost_temp = np.zeros(10)
    training_acc_temp = np.zeros(10)
    for i in range(10):
      training_cost_temp[i], training_acc_temp[i] = sess.run([net.cost, net.accuracy], 
                                                              feed_dict={net.x:train_input[:, i*5000:(i+1)*5000].T, 
                                                                         net.labels:train_output[i*5000:(i+1)*5000],
                                                                         net.keep_prob: 1.0})
    training_cost = np.mean(training_cost_temp)
    training_acc = np.mean(training_acc_temp)

    print("Predicted Image Portion Test Cost %g, Accuracy %g" %(testing_cost, testing_acc))
    print("Predicted Image Portion Training Cost %g, Accuracy %g" %(training_cost, training_acc))

    # Save the model
    modelStr = "/tmp/model_cifar_classification.ckpt"
    save_path = net.saver.save(sess, os.getcwd() + modelStr)
    print("Model saved in file: %s" % save_path)

def train(maskVecXoneYzero, squareSideLength, nCols, bottleneck, n_iterations=20000):
  train, test, train_labels, test_labels = cifar.returnCIFARdata()

  #train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
  #test_hideRight, Xtest_hideRight, Ytest_hideRight = cifar.returnHalfData(ncols=nCols)

  train_hideCenter, Xtrain_hideCenter, Ytrain_hideCenter, \
  test_hideCenter, Xtest_hideCenter, Ytest_hideCenter = cifar.returnSquareData(sideLength=squareSideLength)

  idxs = np.arange(50000)

  train_input = train_hideCenter
  train_output = train

  test_input = test_hideCenter
  test_output = test

  tf.reset_default_graph()
  net = cifar_tf.create_autoencoder(maskVecXoneYzero=maskVecXoneYzero, bottleneck=bottleneck)

  # initialize things to return
  keepTraining = False
  testing_cost = -1.0
  training_cost = -1.0
  mses_testSplits = np.zeros(10)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    if (False):
      # Restore variables from disk.
      #net.saver.restore(sess, os.getcwd() + "/tmp/model_cifar_ncols_%d.ckpt" %nCols)
      net.saver.restore(sess, os.getcwd() + "/tmp/model_cifar_square_%d.ckpt" %squareSideLength)
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

      # Train and learn :D
      sess.run(net.train_step, feed_dict={net.x: batch_in,
                                          net.y: batch_out,
                                          net.keep_prob: 0.5})

      # Every 100 iterations print out a status
      if i%100 == 0:
        train_cost = sess.run(net.cost, feed_dict={net.x: batch_in,
                                                   net.y: batch_out, 
                                                   net.keep_prob: 1.0})
        print("step %d, batch avg l2 %g" %(i, train_cost))


      # Check if we're learning by 20000 iterations
      if (i==10000):
        train_cost = sess.run(net.cost, feed_dict={net.x: batch_in,
                                                   net.y: batch_out, 
                                                   net.keep_prob: 1.0})
        if ((train_cost > 0.08)):
          print("Not training :(")
          keepTraining = True
          break

    # Test and Training Accuracies
    testing_cost_temp = np.zeros(2)
    testing_acc_temp = np.zeros(2)
    for i in range(2):
      testing_cost_temp[i] = sess.run(net.cost, feed_dict={net.x: test_input[:, i*5000:(i+1)*5000].T,
                                                           net.y: test_output[:, i*5000:(i+1)*5000].T,
                                                           net.keep_prob: 1.0})
    testing_cost = np.mean(testing_cost_temp)
    testing_acc = np.mean(testing_acc_temp)

    training_cost_temp = np.zeros(10)
    training_acc_temp = np.zeros(10)
    for i in range(10):
      training_cost_temp[i] = sess.run(net.cost, feed_dict={net.x:train_input[:, i*5000:(i+1)*5000].T, 
                                                            net.y:train_output[:, i*5000:(i+1)*5000].T,
                                                            net.keep_prob: 1.0})
    training_cost = np.mean(training_cost_temp)
    print("Predicted Image Portion Test Cost %g" %testing_cost)
    print("Predicted Image Portion Training Cost %g" %training_cost) 

    # If this was a succesful train...
    if (not keepTraining):
      # Save the model
      #modelStr = "/tmp/model_cifar_ncols_%d.ckpt" %nCols
      modelStr = "/tmp/model_cifar_square_%d.ckpt" %squareSideLength
      save_path = net.saver.save(sess, os.getcwd() + modelStr)
      print("Model saved in file: %s" % save_path)

      # Get splits of test set
      testSplit_in = np.zeros((10, test_input.shape[0], test_input.shape[1]/10))     # input to the conv net (hidden image)
      testSplit_out = np.zeros((10, test_output.shape[0], test_output.shape[1]/10))  # ground truth of conv net output
      for i in range(10):
        testSplit_in[i, :, :] = test_input[:, i*1000:(i+1)*1000]
        testSplit_out[i, :, :] = test_output[:, i*1000:(i+1)*1000]

        predicted_testSplit = sess.run(net.y_conv, feed_dict={net.x:testSplit_in[i].T, 
                                                            net.y:testSplit_out[i].T,
                                                            net.keep_prob: 1.0})

        testSplit_hat_hidden, testSplit_hat_X, testSplit_hat_Y = cifar.hideData(predicted_testSplit.T, maskVecXoneYzero)
        testSplit_hidden, testSplit_X, testSplit_Y = cifar.hideData(testSplit_out[i], maskVecXoneYzero)

        diff_testSplit = testSplit_hat_Y - testSplit_Y
        mses_testSplits[i] = np.mean(np.multiply(diff_testSplit, diff_testSplit))

    return keepTraining, testing_cost, training_cost, mses_testSplits

def train_adversarial(squareSideLength, nCols, bottleneck, n_iterations=20000):
  train, test, train_labels, test_labels = cifar.returnCIFARdata()

  #train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
  #test_hideRight, Xtest_hideRight, Ytest_hideRight = cifar.returnHalfData(ncols=nCols)

  train_hideCenter, Xtrain_hideCenter, Ytrain_hideCenter, \
  test_hideCenter, Xtest_hideCenter, Ytest_hideCenter = cifar.returnSquareData(sideLength=squareSideLength)

  idxs = np.arange(50000)

  train_input = train_hideCenter
  train_output = Ytrain_hideCenter

  test_input = test_hideCenter
  test_output = Ytest_hideCenter

  tf.reset_default_graph()
  net = cifar_tf.create_autoencoder_adversarial(bottleneck=bottleneck)

  # initialize things to return
  keepTraining = False
  testing_cost = -1.0
  training_cost = -1.0
  mses_testSplits = np.zeros(10)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    if (False):
      # Restore variables from disk.
      #net.saver.restore(sess, os.getcwd() + "/tmp/model_cifar_ncols_%d.ckpt" %nCols)
      net.saver.restore(sess, os.getcwd() + "/tmp/model_cifar_square_%d.ckpt" %squareSideLength)
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

      # Train and learn :D
      if (i < 10000):
         sess.run(net.train_step, feed_dict={net.x: batch_in,
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
        if (i < 10000):
          train_cost = sess.run(net.cost, feed_dict={net.x: batch_in,
                                                      net.y: batch_out, 
                                                      net.keep_prob: 1.0})
          print("step %d, batch avg l2 %g" %(i, train_cost))
        else:
          train_cost, train_d_loss, disc_fake, disc_real = sess.run([net.cost, net.d_loss, net.avg_disc_fake, net.avg_disc_real], feed_dict={net.x: batch_in,
                                                     net.y: batch_out, 
                                                     net.keep_prob: 1.0})
          print("step %d, l2 %g, adv %g" %(i, train_cost, train_d_loss))
          print("*****disc_fake %g, disc_real %g" %(disc_fake, disc_real))

      # Check if we're learning by 20000 iterations
      if (i==9000):
        train_cost = sess.run(net.cost, feed_dict={net.x: batch_in,
                                                   net.y: batch_out, 
                                                   net.keep_prob: 1.0})
        if ((train_cost > 0.08)):
          print("Not training :(")
          keepTraining = True
          break

    # Test and Training Accuracies
    testing_cost_temp = np.zeros(2)
    testing_acc_temp = np.zeros(2)
    for i in range(2):
      testing_cost_temp[i] = sess.run(net.cost, feed_dict={net.x: test_input[:, i*5000:(i+1)*5000].T,
                                                           net.y: test_output[:, i*5000:(i+1)*5000].T,
                                                           net.keep_prob: 1.0})
    testing_cost = np.mean(testing_cost_temp)
    testing_acc = np.mean(testing_acc_temp)

    training_cost_temp = np.zeros(10)
    training_acc_temp = np.zeros(10)
    for i in range(10):
      training_cost_temp[i] = sess.run(net.cost, feed_dict={net.x:train_input[:, i*5000:(i+1)*5000].T, 
                                                            net.y:train_output[:, i*5000:(i+1)*5000].T,
                                                            net.keep_prob: 1.0})
    training_cost = np.mean(training_cost_temp)
    print("Predicted Image Portion Test Cost %g" %testing_cost)
    print("Predicted Image Portion Training Cost %g" %training_cost) 

    # If this was a succesful train...
    if (not keepTraining):
      # Save the model
      #modelStr = "/tmp/model_cifar_ncols_%d.ckpt" %nCols
      #modelStr = "/tmp/model_cifar_square_%d.ckpt" %squareSideLength
      modelStr = "/tmp/model_cifar_square_adversarial_%d.ckpt" %squareSideLength
      save_path = net.saver.save(sess, os.getcwd() + modelStr)
      print("Model saved in file: %s" % save_path)

    return keepTraining, testing_cost, training_cost


#train_classification(bottleneck=512)

numTrials = 1

test_costs_data = np.zeros((numTrials))
train_costs_data = np.zeros((numTrials))
mses_testSplits_data = np.zeros((numTrials, 10))
waitedTooLong = np.zeros(numTrials)
# all columns: np.arange(1, 33)
for i in range(numTrials):
  keepTraining = True

  countIdx = 0
  while (keepTraining):
    #nCols = (i+1)*4
    #squareSideLength = 0
    #vecMask = cifar.generateColumnMask(nCols)
    
    nCols = 0
    squareSideLength = 16 
    vecMask = cifar.generateCenterSquareMask(squareSideLength)
    bottleneck = 768

    #print '------------Columns Removed: %d -------------' %nCols
    print '------------Square Size: %d x %d -------------' %(squareSideLength , squareSideLength)

    #keepTraining, test_costs_data[i], train_costs_data[i], \
    #mses_testSplits_data[i] = train(maskVecXoneYzero= vecMask, squareSideLength=squareSideLength, nCols = nCols, bottleneck=bottleneck)

    keepTraining, test_costs_data[i], train_costs_data[i] = train_adversarial(squareSideLength=squareSideLength, nCols = nCols, bottleneck=bottleneck)

    countIdx += 1
    if (countIdx > 10):
      waitedTooLong[i] = 1;
      break

  print '----- Test and Train Costs (MSE of generated image, whole dataset ----'
  print test_costs_data
  print train_costs_data
  print '----- Test Splits (MSE of generated image, test splits ----'
  print mses_testSplits_data
  print '----- Waiting for Waiting for Waiting ----'
  print waitedTooLong

# np.save('cifar_data_test_costs_ncols.npy', test_costs_data)
# np.save('cifar_data_train_costs_ncols.npy', train_costs_data)
# np.save('cifar_data_mses_testSplits_ncols.npy', mses_testSplits_data)

# np.save('cifar_data_test_costs_square.npy', test_costs_data)
# np.save('cifar_data_train_costs_square.npy', train_costs_data)
# np.save('cifar_data_mses_testSplits_square.npy', mses_testSplits_data)
# print 'results saved'