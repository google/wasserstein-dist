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
"""Class that computes the Wasserstein distance in tensorflow.

   The implementation follows Algorithm 2 in [Genevay Aude, Marco Cuturi,
   Gabriel Peyre, Francis Bach, "Stochastic Optimization for Large-scale
   Optimal Transport", NIPS 2016], which compares a distribution to a
   fixed set of samples. Internally, base distances are recomputed a lot.
   To just compute the Wasserstein distance between to sets of points,
   don't use this code, just do a bipartitle matching.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Wasserstein(object):
  """Class to hold (ref to) data and compute Wasserstein distance."""

  def __init__(self, source_gen, target_gen, basedist=None):
    """Inits Wasserstein with source and target data."""
    self.source_gen = source_gen
    self.source_bs = source_gen.bs
    self.target_gen = target_gen
    self.target_bs = target_gen.bs
    self.gradbs = self.source_bs  # number of source sample to compute gradient
    if basedist is None:
      basedist = self.l2dist
    self.basedist = basedist

  def add_summary_montage(self, images, name, num=9):
    vis_images = tf.split(images[:num], num_or_size_splits=num, axis=0)
    vis_images = tf.concat(vis_images, axis=2)
    tf.summary.image(name, vis_images)
    return vis_images

  def add_summary_images(self, num=9):
    """Visualize source images and nearest neighbors from target."""
    source_ims = self.source_gen.get_batch(bs=num, reuse=True)
    vis_images = self.add_summary_montage(source_ims, 'source_ims', num)

    target_ims = self.target_gen.get_batch()
    _ = self.add_summary_montage(target_ims, 'target_ims', num)

    c_xy = self.basedist(source_ims, target_ims)  # pairwise cost
    idx = tf.argmin(c_xy, axis=1)  # find nearest neighbors
    matches = tf.gather(target_ims, idx)
    vis_matches = self.add_summary_montage(matches, 'neighbors_ims', num)

    vis_both = tf.concat([vis_images, vis_matches], axis=1)
    tf.summary.image('matches_ims', vis_both)

    return

  def l2dist(self, source, target):
    """Computes pairwise Euclidean distances in tensorflow."""
    def flatten_batch(x):
      dim = tf.reduce_prod(tf.shape(x)[1:])
      return tf.reshape(x, [-1, dim])
    def scale_batch(x):
      dim = tf.reduce_prod(tf.shape(x)[1:])
      return x/tf.sqrt(tf.cast(dim, tf.float32))
    def prepare_batch(x):
      return scale_batch(flatten_batch(x))

    target_flat = prepare_batch(target)  # shape: [bs, nt]
    target_sqnorms = tf.reduce_sum(tf.square(target_flat), axis=1, keep_dims=True)
    target_sqnorms_t = tf.transpose(target_sqnorms)

    source_flat = prepare_batch(source)  # shape: [bs, ns]
    source_sqnorms = tf.reduce_sum(tf.square(source_flat), axis=1, keep_dims=True)

    dotprod = tf.matmul(source_flat, target_flat, transpose_b=True)  # [ns, nt]
    sqdist = source_sqnorms - 2*dotprod + target_sqnorms_t
    dist = tf.sqrt(tf.nn.relu(sqdist))  # potential tiny negatives are suppressed
    return dist  # shape: [ns, nt]

  def grad_hbar(self, v, gradbs, reuse=True):
    """Compute gradient of hbar function for Wasserstein iteration."""
    source_ims = self.source_gen.get_batch(bs=gradbs, reuse=reuse)
    target_data = self.target_gen.get_batch()

    c_xy = self.basedist(source_ims, target_data)
    c_xy -= v  # [gradbs, trnsize]
    idx = tf.argmin(c_xy, axis=1)               # [1] (index of subgradient)
    target_bs = self.target_bs
    xi_ij = tf.one_hot(idx, target_bs)  # find matches, [gradbs, trnsize]
    xi_ij = tf.reduce_mean(xi_ij, axis=0, keep_dims=True)    # [1, trnsize]
    grad = 1./target_bs - xi_ij  # output: [1, trnsize]
    return grad

  def hbar(self, v, reuse=True):
    """Compute value of hbar function for Wasserstein iteration."""
    source_ims = self.source_gen.get_batch(bs=None, reuse=reuse)
    target_data = self.target_gen.get_batch()

    c_xy = self.basedist(source_ims, target_data)
    c_avg = tf.reduce_mean(c_xy)
    c_xy -= c_avg
    c_xy -= v

    c_xy_min = tf.reduce_min(c_xy, axis=1)  # min_y[ c(x, y) - v(y) ]
    c_xy_min = tf.reduce_mean(c_xy_min)     # expectation wrt x
    return tf.reduce_mean(v, axis=1) + c_xy_min + c_avg # avg wrt y

  def k_step(self, k, v, vt, c, reuse=True):
    """Perform one update step of Wasserstein computation."""
    grad_h = self.grad_hbar(vt, gradbs=self.gradbs, reuse=reuse)
    vt = tf.assign_add(vt, c/tf.sqrt(k)*grad_h, name='vt_assign_add')
    v = ((k-1.)*v + vt)/k
    return k+1, v, vt, c

  def dist(self, C=.1, nsteps=10, reset=False):
    """Compute Wasserstein distance (Alg.2 in [Genevay etal, NIPS'16])."""
    target_bs = self.target_bs
    vtilde = tf.Variable(tf.zeros([1, target_bs]), name='vtilde')
    v = tf.Variable(tf.zeros([1, target_bs]), name='v')
    k = tf.Variable(1., name='k')

    k = k.assign(1.)  # restart averaging from 1 in each call
    if reset:  # used for randomly sampled target data, otherwise warmstart
      v = v.assign(tf.zeros([1, target_bs]))  # reset every time graph is evaluated
      vtilde = vtilde.assign(tf.zeros([1, target_bs]))

    # (unrolled) optimization loop. first iteration, create variables
    k, v, vtilde, C = self.k_step(k, v, vtilde, C, reuse=False)
    # (unrolled) optimization loop. other iterations, reuse variables
    k, v, vtilde, C = tf.while_loop(cond=lambda k, *_: k < nsteps,
                                            body=self.k_step,
                                            loop_vars=[k, v, vtilde, C])
    v = tf.stop_gradient(v)  # only transmit gradient through cost
    val = self.hbar(v)
    return tf.reduce_mean(val)



