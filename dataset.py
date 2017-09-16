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
"""Class that loads a dataset (currently CIFAR) for tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import numpy as np

import tensorflow as tf


class Dataset(object):
  """Loads CIFAR dataset and gives access to subsets of its images."""

  def __init__(self, bs, filepattern, label=None):
    self.bs = bs
    self.images = self.load_dataset(filepattern, label=label)
    self.shuffle()

  def load_dataset(self, filepattern, label):
    """Load CIFAR data (all or single label) and scale to [-1, 1] range."""
    if label is None:
      all_labels = range(10)
    else:
      all_labels = [label]

    images = []
    for label in all_labels:
      filename = filepattern % label
      with open(filename, 'r') as f:
        images.extend(cPickle.load(f))

    images = (np.asarray(images, np.float32)/255.)*2.-1.  # rescale to [-1., 1.]
    return images

  def shuffle(self):
    np.random.shuffle(self.images)

  def get_batch(self, bs=None, reuse=True):  # pylint: disable=g-unused-argument
    if not bs:
      bs = self.bs
    return tf.constant(self.images[:bs])
