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
"""Example use of Wasserstein distance class: train a generative network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import tensorflow as tf

from dataset import Dataset
from generator import Generator
from wasserstein import Wasserstein

try:
  xrange
except NameError:
  xrange = range

tf.flags.DEFINE_string('logdir', '/tmp/wasserstein',
                       'Directory to put the training logs.')
tf.flags.DEFINE_string('filepattern', '/tmp/cifar10/cifar_train_class_%d.pic',
                       'Filepattern from which to read the dataset.')
tf.flags.DEFINE_integer('batch_size', 1000, 'Batch size of generator.')
tf.flags.DEFINE_integer('target_batch_size', 1000,'Batch size of dataset.')
tf.flags.DEFINE_integer('num_steps', 10000000, 'Number of batches to process.')
tf.flags.DEFINE_integer('noise_dim', 64, 'Dimension of latent (noise) space.')
tf.flags.DEFINE_float('learning_rate', 0.001, 'AdamOptimizer learning rate.')
tf.flags.DEFINE_float('momentum', 0.5, 'AdamOptimizer momentum.')

FLAGS = tf.flags.FLAGS


def sys_stdout_flush(string):
  sys.stdout.write(string)
  sys.stdout.flush()


def main(_):
  # set useful shortcuts and initialize timing
  tf.logging.set_verbosity(tf.logging.INFO)
  log_dir = FLAGS.logdir
  start_time = time.time()

  # load dataset
  dataset = Dataset(bs=FLAGS.target_batch_size, filepattern=FLAGS.filepattern)

  with tf.Graph().as_default():
    generator = Generator(FLAGS.batch_size, FLAGS.noise_dim)
    wasserstein = Wasserstein(generator, dataset)

    # create optimization problem to solve (adversarial-free GAN)
    loss = wasserstein.dist(C=0.1, nsteps=10)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    adam = tf.train.AdamOptimizer(FLAGS.learning_rate, FLAGS.momentum)
    train_step = adam.minimize(loss, global_step=global_step)

    # add summaries for tensorboard
    tf.summary.scalar('loss', loss)
    wasserstein.add_summary_images(num=9)
    all_summaries = tf.summary.merge_all()

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    sm = tf.train.SessionManager()
    config = tf.ConfigProto()
    try:
      sess = sm.prepare_session('', init_op=init, saver=saver,
                                config=config, checkpoint_dir=log_dir)
    except tf.errors.InvalidArgumentError:
      tf.logging.info('Cannot load old session. Starting new one.')
      sess = tf.Session(config=config)
      sess.run(init)

    # output graph for early inspection
    test_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    test_writer.flush()

    summary, current_loss = sess.run([all_summaries, loss])
    test_writer.add_summary(summary)
    test_writer.flush()

    sys_stdout_flush('Time to start training %f s\n' % (time.time()-start_time))
    start_time = time.time()
    iteration_time = time.time()

    for i in xrange(FLAGS.num_steps):
      if i%10 == 0:
        sys_stdout_flush('.')
      if i%100 == 0:  # record summaries
        summary, current_loss, step = sess.run([all_summaries, loss, global_step])
        test_writer.add_summary(summary, i)
        test_writer.flush()

        sys_stdout_flush('Step %d[%d] (%s): loss %f ' % (i, step, time.ctime(),
                                                     current_loss))
        sys_stdout_flush('iteration: %f s ' % (time.time()-iteration_time))
        iteration_time = time.time()
        sys_stdout_flush('total: %f s ' % (time.time()-start_time))
        sys_stdout_flush('\n')

      sess.run(train_step)
      if i%1000 == 0:
        sys_stdout_flush('Saving snapshot.\n')
        saver.save(sess, os.path.join(log_dir, 'wasserstein.ckpt'),
                   global_step=global_step)

    saver.save(sess, os.path.join(log_dir, 'wasserstein-final.ckpt'))
    sys_stdout_flush('Done.\n')
    test_writer.close()


if __name__ == '__main__':
  tf.app.run(main)
