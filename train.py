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
  #train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
  #test_hideRight, Xtest_hideRight, Ytest_hideRight = mnist_preprocessing.returnHalfData(ncols=ncols)

  train_hideMiddle, Xtrain_hideMiddle, Ytrain_hideMiddle, \
  test_hideMiddle, Xtest_hideMiddle, Ytest_hideMiddle = mnist_preprocessing.returnSquareData(squareSideLength=squareSideLength)

  idxs = np.arange(10000)

  X_input = train_hideMiddle
  Y_output = train

  test_X_input = test_hideMiddle
  test_Y_output = test

  net = mnist_tf.create_network_autoencoder(bottleneck=bottleneck)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(n_iterations):
      if i%200 == 0:
        np.random.shuffle(idxs)
        bufferedIdxs = np.concatenate((idxs, idxs[0:50]))

      leftIdx = (i*50)%10000
      rightIdx = leftIdx + 50

      batch_X = X_input[:, bufferedIdxs[leftIdx:rightIdx]].T
      batch_Y = Y_output[:, bufferedIdxs[leftIdx:rightIdx]].T

      sess.run(net.train_step, feed_dict={net.x: batch_X,
                                          net.y: batch_Y, 
                                          net.keep_prob: 0.5})

      if i%100 == 0:
        train_cost = sess.run(net.cost, feed_dict={net.x: batch_X,
                                                   net.y: batch_Y, 
                                                   net.keep_prob: 1.0})
        print("step %d, batch avg l2 %g"%(i, train_cost))

      if (i==5000):
        train_cost = sess.run(net.cost, feed_dict={net.x: batch_X,
                                                   net.y: batch_Y, 
                                                   net.keep_prob: 1.0})
        if train_cost > 0.05:
          print("Not training :(")
          break

    # Test and Training Accuracies
    testing_cost = sess.run(net.cost, feed_dict={net.x: test_X_input.T,
                                              net.y: test_Y_output.T, 
                                              net.keep_prob: 1.0})

    training_cost = sess.run(net.cost, feed_dict={net.x:X_input.T, 
                                                  net.y:Y_output.T,
                                                  net.keep_prob: 1.0})

    print("Final Test Cost %g" %testing_cost)
    print("Final Training Cost %g" %training_cost)

    # Save the variables to disk.
    save_path = net.saver.save(sess, os.getcwd() + "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    return testing_cost, training_cost

#train_original()
#train()

'''
numTrials = 1
test_costs = np.zeros(numTrials)
train_costs = np.zeros(numTrials)

for i in range(numTrials):
  test_costs[i], train_costs[i] = train()
  print test_costs
  print train_costs
'''


numTrials = 1
test_costs = np.zeros((numTrials,5))
train_costs = np.zeros((numTrials,5))

for i in range(numTrials):
  for j in range(5):
    sideLength = (i+3)*2
    print '------------ Square: %d x %d -----------' %(sideLength, sideLength)
    test_costs[i][j], train_costs[i][j] = train(squareSideLength=sideLength, ncols = 0, bottleneck=128)
    print test_costs
    print train_costs