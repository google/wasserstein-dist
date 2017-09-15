# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compute Wasserstein distances between different subsets of CIFAR.

   Note: comparing two fixed sets is a sanity check, not the target use case.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import tensorflow as tf

from dataset import Dataset
from wasserstein import Wasserstein

tf.flags.DEFINE_string('filepattern', '/tmp/cifar10/cifar_train_class_%d.pic',
                       'Filepattern from which to read the dataset.')
tf.flags.DEFINE_integer('batch_size', 1000, 'Batch size of generator.')
tf.flags.DEFINE_integer('loss_steps', 50, 'Number of optimization steps.')

FLAGS = tf.flags.FLAGS


def print_flush(string):
  sys.stdout.write(string)
  sys.stdout.flush()


def main(unused_argv):
  # tf.logging.set_verbosity(tf.logging.INFO)

  # load two copies of the dataset
  print('Loading datasets...')
  dataset = [Dataset(bs=FLAGS.batch_size, filepattern=FLAGS.filepattern,
                     label=i) for i in range(10)]

  print('Computing Wasserstein distance(s)...')
  for i in range(10):
    for j in range(10):
      with tf.Graph().as_default():
        # compute Wasserstein distance between sets of labels i and j
        wasserstein = Wasserstein(dataset[i], dataset[j])
        loss = wasserstein.dist(C=.1, nsteps=FLAGS.loss_steps)
        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          res = sess.run(loss)
          print_flush('%f ' % res)
    print_flush('\n')

if __name__ == '__main__':
  tf.app.run(main)
