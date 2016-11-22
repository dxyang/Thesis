import tensorflow as tf
import numpy as np
import os
import mnist_tf
import mnist_preprocessing

train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
test_hideRight, Xtest_hideRight, Ytest_hideRight = mnist_preprocessing.returnData()

def train(n_iterations=20000):
  net = mnist_tf.create_network()
  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter(
                        os.getcwd(), graph=sess.graph)

    sess.run(tf.initialize_all_variables())

    for i in range(n_iterations):
      if (i < 9950):
        leftIdx = i
        rightIdx = (i + 50)
        batch_X = train_hideRight[:, leftIdx:rightIdx].T
        batch_Y = Ytrain_hideRight[:, leftIdx:rightIdx].T
      else:
        randomIdxs = np.random.randint(0, 10000, 50)
        batch_X = train_hideRight[:, randomIdxs].T
        batch_Y = Ytrain_hideRight[:, randomIdxs].T

      sess.run(net.train_step, feed_dict={net.x: batch_X, 
                                          net.y_: batch_Y, 
                                          net.keep_prob: 0.5})

      if i%100 == 0:
        summary = sess.run(net.summary_op, 
                           feed_dict={net.x: batch_X, 
                                      net.y_: batch_Y, 
                                      net.keep_prob: 1.0})
        image_summary = sess.run(net.image_summary_op, 
                                 feed_dict={net.x: batch_X, 
                                            net.y_: batch_Y, 
                                            net.keep_prob: 1.0})

        summary_writer.add_summary(summary, i)
        summary_writer.add_summary(image_summary, i)

        train_mse = sess.run(net.mse, feed_dict={net.x: batch_X, 
                                                net.y_: batch_Y, 
                                                net.keep_prob: 1.0})
        print("step %d, batch average mean square error %g"%(i, train_mse))

    # Test and Training Accuracies
    test_mse = sess.run(net.mse, feed_dict={net.x:test_hideRight.T, 
                                            net.y_:Ytest_hideRight.T, 
                                            net.keep_prob: 1.0})
    print("Final Test MSE %g" %test_mse)
    training_mse = sess.run(net.mse, feed_dict={net.x:train_hideRight.T, 
                                                net.y_:Ytrain_hideRight.T, 
                                                net.keep_prob: 1.0})
    print("Final Training MSE %g" %training_mse)

    # Save the variables to disk.
    save_path = net.saver.save(sess, os.getcwd() + "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

train()