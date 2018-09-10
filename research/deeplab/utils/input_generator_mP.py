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
from deeplab.datasets.regression_dataset_mP import DatasetDescriptor, _DATASETS_INFORMATION, _APOLLOSCAPE_INFORMATION
slim = tf.contrib.slim
import numpy as np
# from deeplab import model
from deeplab import model_twoBranch as model

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

def _get_data(dataset, model_options, data_provider, dataset_split, codes_cons):
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

  if dataset_split != common.TEST_SET:
    image, image_name, vis, height, width, seg, shape_id_map_gt, pose_dict, shape_id_dict = data_provider.get([common.IMAGE, common.IMAGE_NAME, 'vis', common.HEIGHT, common.WIDTH, 'seg', 'shape_id_map', 'pose_dict', 'shape_id_dict'])
    vis = tf.reshape(vis, [dataset.height, dataset.width, 3])
  else:
    image, image_name, height, width, seg = data_provider.get([common.IMAGE, common.IMAGE_NAME, common.HEIGHT, common.WIDTH, 'seg'])

  image = tf.reshape(image, [dataset.height, dataset.width,3])
  seg = tf.reshape(seg, [dataset.height, dataset.width, 1])
  seg_float = tf.cast(seg, tf.float32)
  mask = tf.greater(seg, 0)
  mask_float = tf.cast(mask, tf.float32)
  mask_int32 = tf.cast(mask, tf.int32)
  seg = tf.cast(seg, tf.int32)
  seg_flatten = tf.reshape(seg, [-1])

  logits_output_stride = (
      model_options.decoder_output_stride or model_options.output_stride)
  logits_height = model.scale_dimension(
      tf.shape(image)[0],
      1.0 / logits_output_stride)
  logits_width = model.scale_dimension(
      tf.shape(image)[1],
      1.0 / logits_output_stride)
  seg_rescaled_float = tf.squeeze(
      tf.image.resize_nearest_neighbor(tf.expand_dims(seg_float, 0), [logits_height, logits_width], align_corners=True), 0)
  seg_rescaled_flatten = tf.cast(tf.reshape(seg_rescaled_float, [-1]), tf.int32)
  mask_rescaled_float = tf.squeeze(
      tf.image.resize_nearest_neighbor(tf.expand_dims(mask_float, 0), [logits_height, logits_width], align_corners=True), 0)

  if dataset_split != common.TEST_SET:
    pose_dict = tf.reshape(pose_dict, [-1, 6])
    pose_dict = tf.identity(pose_dict, name='pose_dict_ori')
    # seg_one_hot_posemap = tf.one_hot(tf.reshape(seg, [-1]), depth=tf.shape(pose_dict)[0])
    # pose_map = tf.matmul(seg_one_hot_posemap, pose_dict)
    # pose_map = tf.reshape(pose_map, [dataset.height, dataset.width, 6])

    shape_id_dict = tf.cast(tf.reshape(shape_id_dict, [-1]), tf.int32)
    shape_id_dict = tf.identity(shape_id_dict, name='shape_id_dict')
    shape_dict = tf.gather(codes_cons, tf.clip_by_value(shape_id_dict, 0, 78))
    # # seg_one_hot_shapemap = tf.one_hot(tf.reshape(seg, [-1]), depth=tf.shape(shape_dict)[0])
    # # shape_map = tf.matmul(seg_one_hot_shapemap, shape_dict)
    # shape_map = tf.gather(shape_dict, seg_flatten)
    # shape_map = tf.reshape(shape_map, [dataset.height, dataset.width, dataset.SHAPE_DIMS])
    # # shape_map_masked = tf.where(tf.tile(mask, [1, 1, dataset.SHAPE_DIMS]), shape_map, tf.zeros_like(shape_map))
    # shape_map_masked = tf.multiply(shape_map, mask_float)

    # # shape_id_map = tf.matmul(seg_one_hot_shapemap, tf.reshape(shape_id_dict, [-1, 1]))
    # shape_id_map = tf.gather(tf.reshape(shape_id_dict, [-1, 1]), seg_flatten)
    # shape_id_map = tf.reshape(shape_id_map, [dataset.height, dataset.width, 1])
    # # shape_id_map_masked = tf.where(tf.tile(mask, [1, 1, tf.shape(shape_id_map)[2]]), shape_id_map, tf.zeros_like(shape_id_map))
    # shape_id_map_masked = tf.multiply(shape_id_map, mask_int32)

    # # Getting inverse depth outof the posemap
    # pose_map_depth = tf.gather(pose_map, [5], axis=2)
    # # pose_map_invd = tf.clip_by_value(tf.multiply(1./pose_map_depth, mask_float), 0., 0.25) # 1/2
    # pose_map_invd = tf.multiply(pose_map_depth, mask_float) # 2/2
    # pose_map_angles = tf.gather(pose_map, [0, 1, 2], axis=2)
    # pose_map_quat = euler_angles_to_quaternions(pose_map_angles)
    # pose_map = tf.concat([pose_map_quat, tf.gather(pose_map, [3, 4], axis=2), pose_map_invd], axis=2)
    # pose_map_masked = tf.multiply(pose_map, mask_float)

    # pose_shape_map_masked = tf.concat([pose_map_masked, shape_map_masked], axis=2)

    # returning per-instance pose_dict and shape_dict, and segs
    pose_dict_depth = tf.clip_by_value(tf.gather(pose_dict, [5], axis=1), 0., 300.)
    pose_dict_invd = tf.clip_by_value(tf.reciprocal(pose_dict_depth), 0., 0.25)
    pose_dict_angles = tf.gather(pose_dict, [0, 1, 2], axis=1)
    pose_dict_quat = euler_angles_to_quaternions(pose_dict_angles)
    pose_dict_x = tf.clip_by_value(tf.gather(pose_dict, [3], axis=1), -100., 100.)
    pose_dict_y = tf.clip_by_value(tf.gather(pose_dict, [4], axis=1), 0., 50.)
    # pose_dict_quat_invd = tf.concat([pose_dict_quat, pose_dict_x, pose_dict_y, pose_dict_invd], axis=1) # 1/2: reg invd
    pose_dict_quat_invd = tf.concat([pose_dict_quat, pose_dict_x, pose_dict_y, pose_dict_depth], axis=1) # 2/2: reg depth

    pose_dict_quat_invd = tf.slice(pose_dict_quat_invd, [1, 0], [-1, -1]) # [N, D_p]
    shape_dict = tf.slice(shape_dict, [1, 0], [-1, -1]) # [N, D_s]
    shape_id_dict = tf.cast(tf.slice(tf.expand_dims(shape_id_dict, -1), [1, 0], [-1, -1]), tf.int64) # [N, 1]
    # seg_one_hots_flattened = tf.transpose(tf.one_hot(tf.squeeze(seg_rescaled_flatten), depth=tf.shape(shape_dict)[0])) # [N+1, H*W]
    # seg_one_hots_flattened = tf.cast(tf.slice(seg_one_hots_flattened, [1, 0], [-1, -1]), tf.int32) # [N, H*W]

    idxs = tf.expand_dims(tf.range(tf.shape(shape_dict)[0]), -1) # [N, 1]

    return image, vis, image_name, height, width, seg_rescaled_float, seg_float, mask, mask_rescaled_float, pose_dict_quat_invd, shape_dict, shape_id_dict, idxs
    # return image, vis, pose_map_masked, shape_map_masked, pose_shape_map_masked, shape_id_map_masked, image_name, height, width, seg_rescaled_float, seg_float, mask, mask_rescaled_float, pose_dict_quat_invd, shape_dict, shape_id_dict, seg_one_hots_flattened, idxs
  else:
    return image, image_name, height, width, seg_float, mask


def get(dataset,
    model_options,
        codes,
        batch_size,
        num_readers=1,
        num_threads=1,
        dataset_split=None,
        is_training=True,
        model_variant=None,
        num_epochs=None):
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
  options = {}
  data_provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      reader_kwargs=options,
      num_epochs=num_epochs,
      shuffle=is_training)
  codes_cons = tf.constant(np.transpose(codes), dtype=tf.float32)
  if dataset_split != common.TEST_SET:
    image, vis, image_name, height, width, seg_rescaled, seg, mask, mask_rescaled_float, pose_dict, shape_dict, shape_id_dict, idxs = _get_data(dataset, model_options, data_provider, dataset_split, codes_cons)
    # image, vis, pose_map_masked, shape_map_masked, pose_shape_map_masked, shape_id_map, image_name, height, width, seg_rescaled, seg, mask, mask_rescaled_float, pose_dict, shape_dict, shape_id_dict, seg_one_hots_flattened, idxs = _get_data(dataset, model_options, data_provider, dataset_split, codes_cons)
  else:
    image, image_name, height, width, seg, mask = _get_data(dataset, model_options, data_provider, dataset_split, codes_cons)

  # pose_dict.set_shape([None, 7])
  # shape_dict.set_shape([None, 10])

  if dataset_split != common.TEST_SET:
    sample = {
      common.IMAGE: image,
      'vis': vis,
      common.IMAGE_NAME: image_name,
      common.HEIGHT: height,
      common.WIDTH: width,
      # 'pose_map': pose_map_masked,
      # 'shape_map': shape_map_masked,
      # 'label_pose_shape_map': pose_shape_map_masked,
      # 'shape_id_map': shape_id_map,
      'seg': seg, # float
      'seg_rescaled': seg_rescaled, # float
      'mask': mask,
      'mask_rescaled_float': mask_rescaled_float,
      'pose_dict': pose_dict,
      'shape_dict': shape_dict,
      'shape_id_dict': shape_id_dict,
      # 'seg_one_hots_flattened': seg_one_hots_flattened, # int32
      'idxs': idxs,
      'car_nums': tf.shape(pose_dict)[0]}
  else:
    sample = {
      common.IMAGE: image,
      common.IMAGE_NAME: image_name,
      common.HEIGHT: height,
      common.WIDTH: width,
      'seg': seg,
      'mask': mask}

  return tf.train.batch(
      sample,
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=60 if num_epochs==None else batch_size,
      allow_smaller_final_batch=False,
      dynamic_pad=True)
