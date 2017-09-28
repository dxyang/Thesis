import os
import numpy as np
import tensorflow as tf
import time

from sklearn.model_selection import train_test_split
from ops import *
from utils import *
from glob import glob

class DCGAN(object):
  def __init__(self, sess):
    """
    Args:
      sess: TensorFlow session
    """
    self.sess = sess
    self.batch_size = 64
    self.z_dim = 100

    self.learning_rate = 0.0002
    self.beta1 = 0.5
    self.epoch = 25
    self.input_box = 108 # changes the box around which you rescale the input img
    self.output_box = 64 # 64 x 64 x 3 imgs

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')

    self.build_model()


  def build_model(self):
    self.inputs = tf.placeholder(tf.float32, [None, 64, 64, 3], name='real_images')
    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

    self.G = self.generator(self.z)
    self.D, self.D_logits = self.discriminator(self.inputs)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

    self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, targets=tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, targets=tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, targets=tf.ones_like(self.D_)))
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    # for image completion
    self.mask = tf.placeholder(tf.float32, [None, 64, 64, 3], name='mask')
    self.inverseMask = tf.placeholder(tf.float32, [None, 64, 64, 3], name='mask')
    self.hiddenImg_fake = tf.mul(self.mask, self.G)
    self.hiddenImg_real = tf.mul(self.mask, self.inputs)
    self.restorationSection = tf.mul(self.inverseMask, self.G)
    self.reconstruction = tf.add(self.hiddenImg_real, self.restorationSection)

    self.contextual_loss = tf.reduce_sum(
        tf.contrib.layers.flatten(
            tf.abs(self.hiddenImg_real - self.hiddenImg_fake)), 1)
    self.perceptual_loss = self.g_loss
    self.complete_loss = self.contextual_loss + self.perceptual_loss
    self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

    self.saver = tf.train.Saver()

  def discriminator(self, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, 64, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, 128, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, 256, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, 512, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, 4*4*512]), 1, 'd_h3_lin')

      return tf.nn.sigmoid(h4), h4

  def generator(self, z, reuse=False):
    with tf.variable_scope("generator") as scope:
      if reuse:
        scope.reuse_variables()

      # processing batch size (normally 64 for train but varies
      # for evaluating on different inputs)
      batchSize = tf.shape(z)[0]
      print batchSize

      # project `z` and reshape
      self.z_, self.h0_w, self.h0_b = linear(
        z, 4*4*512, 'g_h0_lin', with_w=True)

      self.h0 = tf.reshape(self.z_, [-1, 4, 4, 512])
      h0 = tf.nn.relu(self.g_bn0(self.h0))

      self.h1, self.h1_w, self.h1_b = deconv2d(
          h0, [batchSize, 8, 8, 256], name='g_h1', with_w=True)
      h1 = tf.nn.relu(self.g_bn1(self.h1))

      h2, self.h2_w, self.h2_b = deconv2d(
          h1, [batchSize, 16, 16, 128], name='g_h2', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3, self.h3_w, self.h3_b = deconv2d(
          h2, [batchSize, 32, 32, 64], name='g_h3', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(h3))

      h4, self.h4_w, self.h4_b = deconv2d(
          h3, [batchSize, 64, 64, 3], name='g_h4', with_w=True)

      return tf.nn.tanh(h4)

  def predict(self):
    constant_z = np.load("constant_z_64x100.npy")
    genImgs = self.sess.run(self.G,
                feed_dict={ self.z: constant_z })
    np.save('predictedFaces.npy', genImgs)
    save_images(genImgs, [8, 8], 'predictedFaces.png')

  def predict_duringTrain(self, epoch, counter):
    # constant_z = np.random.uniform(-1, 1, [64, self.z_dim]).astype(np.float32)
    # np.save("constant_z_64x100.npy", constant_z)
    constant_z = np.load("constant_z_64x100.npy")
    genImgs = self.sess.run(self.G,
                feed_dict={ self.z: constant_z })

    np.save('celebA_datasetSplit_April20/predictedFaces_%d_%d.npy' %(epoch, counter), genImgs)
    save_images(genImgs, [8, 8], 'test_April20_%d_%d.png' %(epoch, counter))

  def imageCompletion(self):
    bs = 64
    momentum = 0.9
    lr = 0.01

    data = glob(os.path.join("./data", "lfw", "*", "*.jpg"))
    dataset = "lfw"
    # data = glob(os.path.join("./data", "celebA", "*.jpg"))
    # dataset = "celebA"

    # train = np.load("train_imgPaths.npy")
    # test = np.load("test_imgPaths.npy")

    # data = test
    np.random.shuffle(data)
    print len(data)

    batch_idxs = len(data) // bs

    start_time = time.time()
    for idx in np.arange(0, 20):
      print 'Completing batch %3d, time: %4.4f' %(idx, time.time() - start_time)
      batch_files = data[idx*bs:(idx+1)*bs]
      batch = [
          get_image(batch_file,
                    input_height=114, #108
                    input_width=114, #108
                    resize_height=64,
                    resize_width=64,
                    is_crop=True) for batch_file in batch_files]

      # images to hide middle from
      batch_images = np.array(batch).astype(np.float32)
      
      # noise vectors to optimize
      zhats = np.random.uniform(-1, 1, size=(bs, self.z_dim))

      # center mask
      mask = np.ones((64, 64, 3))
      left = int(0.25*64) 
      right = int(0.75*64)
      mask[left:right, left:right, :] = 0
      invMask = np.ones((64, 64, 3)) - mask
      maskMatrix = np.resize(mask, [bs, 64, 64, 3])
      invMaskMatrix = np.resize(invMask, [bs, 64, 64, 3])

      v = 0
      for i in range(1000):
          fd = {
              self.z: zhats,
              self.mask: maskMatrix,
              self.inverseMask: invMaskMatrix,
              self.inputs: batch_images
          }
          run = [self.complete_loss, self.grad_complete_loss, self.G, self.reconstruction]
          loss, g, G_imgs, recon = self.sess.run(run, feed_dict=fd)

          if (i == 0):
            hiddenImgs = self.sess.run(self.hiddenImg_real, feed_dict=fd)
            save_images(batch_images, [8, 8], 'BatchGenerations_LFW/img_%s_%d_original.png' %(dataset, idx))
            save_images(hiddenImgs, [8, 8], 'BatchGenerations_LFW/img_%s_%d_originalHidden.png' %(dataset, idx))

          # if (i % 1000) == 0:
          #   save_images(G_imgs, [8, 8], 'img_match_%s_%d_%d.png' %(dataset, idx, i))
          #   save_images(recon, [8, 8], 'img_reconstruction_%s_%d_%d.png' %(dataset, idx, i))

          v_prev = np.copy(v)
          v = momentum*v - lr*g[0]
          zhats += -momentum * v_prev + (1+momentum)*v
          zhats = np.clip(zhats, -1, 1)
      save_images(G_imgs, [8, 8], 'BatchGenerations_LFW/img_%s_%d_%d_match.png' %(dataset, idx, i))
      save_images(recon, [8, 8], 'BatchGenerations_LFW/img_%s_%d_%d_reconstruction.png' %(dataset, idx, i))


  def train(self):
    """Train DCGAN"""
    d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    
    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()
    could_load, checkpoint_counter = self.load("checkpoint")
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
      print "counter %d" %counter
    else:
      print(" [!] Load failed...")
    
    data = glob(os.path.join("./data", "celebA", "*.jpg"))

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    # train_imgPaths = np.save("train_imgPaths.npy", train)
    # test_imgPaths = np.save("test_imgPaths.npy", test)
    # print "saved train test img path split"

    train = np.load("train_imgPaths.npy")
    test = np.load("test_imgPaths.npy")

    data = train
    print len(data)

    batch_idxs = len(data) // self.batch_size
    completedEpochs = counter // batch_idxs
    for epoch in np.arange(completedEpochs, self.epoch):
      if (counter % batch_idxs) == 0:
        start = 0
        self.predict_duringTrain(epoch, counter)
      else:
        start = counter % batch_idxs

      for idx in np.arange(start, batch_idxs):
        batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = [
            get_image(batch_file,
                      input_height=self.input_box,
                      input_width=self.input_box,
                      resize_height=self.output_box,
                      resize_width=self.output_box,
                      is_crop=True) for batch_file in batch_files]
        batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        # Update D network
        self.sess.run([d_optim],
          feed_dict={ self.inputs: batch_images, self.z: batch_z })

        # Update G network
        self.sess.run([g_optim],
          feed_dict={ self.z: batch_z })

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        self.sess.run([g_optim],
          feed_dict={ self.z: batch_z })
        
        errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
        errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
        errG = self.g_loss.eval({self.z: batch_z})

        counter += 1
        print("Counter:[%2d] Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (counter, epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        if np.mod(counter, 500) == 2:
          self.save("checkpoint", counter)

        if (counter % 100) == 0:
          self.predict_duringTrain(epoch, counter)


  @property
  def model_dir(self):
    return "celebA_64_64_64_withSplit"
    #return "celebA_64_64_64_noSplit"

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)
