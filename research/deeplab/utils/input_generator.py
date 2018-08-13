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
"""Wrapper for providing semantic segmentation data."""

import tensorflow as tf
from deeplab import common
from deeplab import input_preprocess
from deeplab.datasets.regression_dataset import DatasetDescriptor, _DATASETS_INFORMATION, _APOLLOSCAPE_INFORMATION
slim = tf.contrib.slim
import numpy as np

dataset_data_provider = slim.dataset_data_provider

def euler_angles_to_quaternions(angle):
    """Convert euler angels to quaternions.
    Input:
    angle: [roll, pitch, yaw]
    """
    roll = tf.gather(angle, [0], axis=-1)
    pitch = tf.gather(angle,[1], axis=-1)
    yaw = tf.gather(angle, [2], axis=-1)

    cy = tf.cos(yaw * 0.5)
    sy = tf.sin(yaw * 0.5)
    cr = tf.cos(roll * 0.5)
    sr = tf.sin(roll * 0.5)
    cp = tf.cos(pitch * 0.5)
    sp = tf.sin(pitch * 0.5)

    q0 = cy * cr * cp + sy * sr * sp
    q1 = cy * sr * cp - sy * cr * sp
    q2 = cy * cr * sp + sy * sr * cp
    q3 = sy * cr * cp - cy * sr * sp
    q = tf.concat([q0, q1, q2, q3], axis=-1)
    return q

def _get_data(dataset, data_provider, dataset_split):
  """Gets data from data provider.

  Args:
    data_provider: An object of slim.data_provider.
    dataset_split: Dataset split.

  Returns:
    image: Image Tensor.
    label: Label Tensor storing segmentation annotations.
    image_name: Image name.
    height: Image height.
    width: Image width.

  Raises:
    ValueError: Failed to find label.
  """
  # if common.LABELS_CLASS not in data_provider.list_items():
  if 'seg' not in data_provider.list_items():
    raise ValueError('Failed to find labels.')

  image, vis, height, width = data_provider.get(
      [common.IMAGE, 'vis', common.HEIGHT, common.WIDTH])
  image = tf.reshape(image, [_DATASETS_INFORMATION[dataset.name].height, _DATASETS_INFORMATION[dataset.name].width,3])
  vis = tf.reshape(vis, [_DATASETS_INFORMATION[dataset.name].height, _DATASETS_INFORMATION[dataset.name].width,3])

  # Some datasets do not contain image_name.
  if common.IMAGE_NAME in data_provider.list_items():
    image_name, = data_provider.get([common.IMAGE_NAME])
  else:
    image_name = tf.constant('')

  label = None
  if dataset_split != common.TEST_SET:
    seg, pose_dict = data_provider.get(['seg', 'pose_dict'])
  if dataset.name == 'apolloscape':
    pose_dict = tf.reshape(pose_dict, [-1, 6])
    pose_dict = tf.identity(pose_dict, name='pose_dict')
    seg = tf.reshape(seg, [_DATASETS_INFORMATION[dataset.name].height, _DATASETS_INFORMATION[dataset.name].width, 1])
    seg_one_hot = tf.one_hot(tf.reshape(seg, [-1]), depth=tf.shape(pose_dict)[0])
    pose_map = tf.matmul(seg_one_hot, pose_dict)
    pose_map = tf.reshape(pose_map, [_DATASETS_INFORMATION[dataset.name].height, _DATASETS_INFORMATION[dataset.name].width, 6])

    seg = tf.cast(seg, tf.float32)
    mask = tf.not_equal(seg, 0.)

    # ## Getting inverse depth outof the posemap
    label_depth = tf.gather(pose_map, [5], axis=2)
    label_invd = tf.where(mask, 1./label_depth, tf.zeros_like(label_depth))
    # label = tf.tile(label_invd, [1, 1, 6])
    # label = tf.concat([tf.gather(pose_map, [0, 1, 2, 3, 4], axis=2), label_invd], axis=2)
    label_angles = tf.gather(pose_map, [0, 1, 2], axis=2)
    label_quat = euler_angles_to_quaternions(label_angles)
    label = tf.concat([label_quat, tf.gather(pose_map, [3, 4], axis=2), label_invd], axis=2)
    label_masked = tf.where(tf.tile(mask, [1, 1, dataset.num_classes]), label, tf.zeros_like(label))


  return image, vis, label_masked, image_name, height, width, seg, mask


def get(dataset,
        crop_size,
        batch_size,
        min_resize_value=None,
        max_resize_value=None,
        resize_factor=None,
        min_scale_factor=1.,
        max_scale_factor=1.,
        scale_factor_step_size=0,
        num_readers=1,
        num_threads=1,
        dataset_split=None,
        is_training=True,
        model_variant=None):
  """Gets the dataset split for semantic segmentation.

  This functions gets the dataset split for semantic segmentation. In
  particular, it is a wrapper of (1) dataset_data_provider which returns the raw
  dataset split, (2) input_preprcess which preprocess the raw data, and (3) the
  Tset_data_provider.DatasetDataProviderensorflow operation of batching the preprocessed data. Then, the output could
  be directly used by training, evaluation or visualization.

  Args:
    dataset: An instance of slim Dataset.
    crop_size: Image crop size [height, width].
    batch_size: Batch size.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    num_readers: Number of readers for data provider.
    num_threads: Number of threads for batching data.
    dataset_split: Dataset split.
    is_training: Is training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.

  Returns:
    A dictionary of batched Tensors for semantic segmentation.

  Raises:
    ValueError: dataset_split is None, failed to find labels, or label shape
      is not valid.
  """
  if dataset_split is None:
    raise ValueError('Unknown dataset split.')
  if model_variant is None:
    tf.logging.warning('Please specify a model_variant. See '
                       'feature_extractor.network_map for supported model '
                       'variants.')
  # options = {'options': tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)}
  options = {}
  data_provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      reader_kwargs=options,
      num_epochs=None,
      shuffle=is_training)
  image, vis, label, image_name, height, width, seg, mask = _get_data(dataset, data_provider,
                                                      dataset_split)
  if label is not None:
    if label.shape.ndims == 2:
      label = tf.expand_dims(label, 2)
    # elif label.shape.ndims == 3 and label.shape.dims[2] == 1:
    elif label.shape.ndims == 3 and label.shape.dims[2] in [1, 6, 3, 7]: # 1 for segmentation label maps, and 6 for posemaps
      pass
    else:
      raise ValueError('Input label shape must be [height, width], or '
                       '[height, width, {1,6}].')
  label.set_shape([None, None, dataset.num_classes])
  sample = {
      common.IMAGE: image,
      'vis': vis,
      common.IMAGE_NAME: image_name,
      common.HEIGHT: height,
      common.WIDTH: width
  }
  if label is not None:
    sample[common.LABEL] = label
    # sample['label_id'] = label_id,
    sample['seg'] = seg
    sample['mask'] = mask

  # if not is_training:
  #   # Original image is only used during visualization.
  #   sample[common.ORIGINAL_IMAGE] = original_image,
  #   num_threads = 1

  return tf.train.batch(
      sample,
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=32 * batch_size,
      allow_smaller_final_batch=False,
      dynamic_pad=False)
