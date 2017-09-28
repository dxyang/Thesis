from model import DCGAN
from utils import show_all_variables

import tensorflow as tf

isTrain = 0

with tf.Session() as sess:
  dcgan = DCGAN(sess)

  show_all_variables()
  if (isTrain):
    dcgan.train()
  else:
    if not dcgan.load("checkpoint"):
      raise Exception("[!] Train a model first, then run test mode")
    #dcgan.predict()
    dcgan.imageCompletion()