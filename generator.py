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
"""Class that generates images using a DCGAN-like network in tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

class Generator(object):
  """Generates images from noise using a DCGAN-like network."""

  def __init__(self, bs, noise_dim):
    self.bs = bs  # batch size
    self.noise_dim = noise_dim  # dimension of latent space

  # minimal implementation of DCGAN-like network for 32x32x3 images
  def network(self, inputs, reuse=True):
    with tf.variable_scope('Generator', values=[inputs], reuse=reuse) as scope:
      with slim.arg_scope([slim.conv2d_transpose], stride=2, kernel_size=4,
                          normalizer_fn=slim.batch_norm):
        net = tf.expand_dims(tf.expand_dims(inputs, 1), 1)
        net = slim.conv2d_transpose(net, 512, stride=1, padding='VALID',
                                    scope='deconv1')
        net = slim.conv2d_transpose(net, 256, scope='deconv2')
        net = slim.conv2d_transpose(net, 128, scope='deconv3')
        net = slim.conv2d_transpose(net, 64, normalizer_fn=None,
                                    activation_fn=None, scope='deconv4')
        logits = slim.conv2d(net, 3, normalizer_fn=None, activation_fn=None,
                             kernel_size=1, stride=1, padding='VALID',
                             scope='logits')
        images = tf.tanh(logits)
        return images

  # generate a batch of images from random noise
  def get_batch(self, bs=None, reuse=True):
    if not bs:
      bs = self.bs
    noise = tf.random_normal([bs, self.noise_dim], name='noise')
    images = self.network(noise, reuse=reuse)
    return images
