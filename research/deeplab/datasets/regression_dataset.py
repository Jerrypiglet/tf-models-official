# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Provides data from semantic regression datasets.

The SegmentationDataset class provides both images and annotations (semantic
segmentation and/or instance segmentation) for TensorFlow. Currently, we
support the following datasets:

1. PASCAL VOC 2012 (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

PASCAL VOC 2012 semantic segmentation dataset annotates 20 foreground objects
(e.g., bike, person, and so on) and leaves all the other semantic classes as
one background class. The dataset contains 1464, 1449, and 1456 annotated
images for the training, validation and test respectively.

2. Cityscapes dataset (https://www.cityscapes-dataset.com)

The Cityscapes dataset contains 19 semantic labels (such as road, person, car,
and so on) for urban street scenes.

3. ADE20K dataset (http://groups.csail.mit.edu/vision/datasets/ADE20K)

The ADE20K dataset contains 150 semantic labels both urban street scenes and
indoor scenes.

References:
  M. Everingham, S. M. A. Eslami, L. V. Gool, C. K. I. Williams, J. Winn,
  and A. Zisserman, The pascal visual object classes challenge a retrospective.
  IJCV, 2014.

  M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
  U. Franke, S. Roth, and B. Schiele, "The cityscapes dataset for semantic urban
  scene understanding," In Proc. of CVPR, 2016.

  B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, A. Torralba, "Scene Parsing
  through ADE20K dataset", In Proc. of CVPR, 2017.
"""
import collections
import os.path
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic regression label whose size matches image.'
                     'Its values range from 0 (background) to real values (pose and shape representations).'),
}

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    ['splits_to_sizes',   # Splits of the dataset into training, val, and test.
     'num_classes',   # Number of semantic classes, including the background
                      # class (if exists). For example, there are 20
                      # foreground classes + 1 background class in the PASCAL
                      # VOC 2012 dataset. Thus, we set num_classes=21.
     # 'ignore_label',  # Ignore label value.
     'height',
     'width',
     'pose_range',
     'bin_nums',
     'space_names',
    ]
)

_APOLLOSCAPE_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 3928,
        'val': 207,
        # 'train': 731,
        # 'val': 107,
    },
    num_classes=6,
    # ignore_label=255.,
    # height=544,
    height=272,
    width=680,
    # pose_range = [[-5.0, 10.0],
    #     [-5.0, 10.0],
    #     [-10.0, 10.0]],
    # bin_nums = [16, 16, 64],
    # space_names = ['x', 'y', 'z'],
    pose_range = [[-np.pi/4, np.pi/4],
        [-np.pi/4, np.pi/4],
        [-np.pi, np.pi],
        [-5.0, 10.0],
        [-5.0, 10.0],
        [-10.0, 10.0]],
    bin_nums = [8, 8, 64, 16, 16, 64],
    space_names = ['row', 'pitch', 'yaw', 'x', 'y', 'z'],
)

_DATASETS_INFORMATION = {
    'apolloscape': _APOLLOSCAPE_INFORMATION,
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'

def get_dataset(dataset_name, split_name, dataset_dir):
  """Gets an instance of slim Dataset.

  Args:
    dataset_name: Dataset name.
    split_name: A train/val Split name.
    dataset_dir: The directory of the dataset sources.

  Returns:
    An instance of slim Dataset.

  Raises:
    ValueError: if the dataset_name or split_name is not recognized.
  """
  if dataset_name not in _DATASETS_INFORMATION:
    raise ValueError('The specified dataset is not supported yet.')

  splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

  if split_name not in splits_to_sizes:
    raise ValueError('data split name %s not recognized' % split_name)

  # Prepare the variables for different datasets.
  num_classes = _DATASETS_INFORMATION[dataset_name].num_classes
  pose_range = _DATASETS_INFORMATION[dataset_name].pose_range
  bin_nums = _DATASETS_INFORMATION[dataset_name].bin_nums
  space_names = _DATASETS_INFORMATION[dataset_name].space_names
  # ignore_label = _DATASETS_INFORMATION[dataset_name].ignore_label

  file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Specify how the TF-Examples are **decoded**.
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/filename': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='png'),
      'image/height': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      # 'image/posemap/class/encoded': tf.VarLenFeature(dtype=tf.float32),
      'posedict/encoded': tf.VarLenFeature(dtype=tf.float32),
      'vis/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'vis/format': tf.FixedLenFeature(
          (), tf.string, default_value='png'),
      'seg/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'seg/format': tf.FixedLenFeature(
          (), tf.string, default_value='png'),
  }
  items_to_handlers = {
      'image': tfexample_decoder.Image(
          image_key='image/encoded',
          format_key='image/format',
          channels=3),
      'vis': tfexample_decoder.Image(
          image_key='vis/encoded',
          format_key='vis/format',
          channels=3),
      'seg': tfexample_decoder.Image(
          image_key='seg/encoded',
          format_key='seg/format',
          channels=1),
      'image_name': tfexample_decoder.Tensor('image/filename'),
      'height': tfexample_decoder.Tensor('image/height'),
      'width': tfexample_decoder.Tensor('image/width'),
      # 'labels_class': tfexample_decoder.Tensor('image/posemap/class/encoded')
      'pose_dict': tfexample_decoder.Tensor('posedict/encoded')
  }

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)
  return dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=splits_to_sizes[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      # ignore_label=ignore_label,
      pose_range=pose_range,
      bin_nums=bin_nums,
      space_names=space_names,
      num_classes=num_classes,
      name=dataset_name,
      multi_label=True)
