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
#  ==============================================================================
"""Utility functions for training."""

import six

import tensorflow as tf
import numpy as np
from deeplab.core import preprocess_utils

slim = tf.contrib.slim

def get_avg_tensor_from_scopes(num_clones, pattern_train_postfix, graph, config, tensor_name):
    tensor_list = []
    for clone_idx in range(num_clones):
      clone_scope = config.clone_scope(clone_idx) # clone_0
      if num_clones > 1:
          pattern_train = clone_scope + '/%s'%pattern_train_postfix
      else:
          pattern_train = pattern_train_postfix
      summary_tensor = graph.get_tensor_by_name(pattern_train%tensor_name)
      tensor_list.append(summary_tensor)
    return tf.add_n(tensor_list)/ num_clones

def scale_logits_to_labels(logits, labels, upsample_logits):
    """ Scaled logits and labels to the same scale."""
    if upsample_logits:
      # Label is not downsampled, and instead we upsample logits.
      scaled_logits = tf.image.resize_bilinear(
            logits,
            preprocess_utils.resolve_shape(labels, 4)[1:3],
            align_corners=True)
      scaled_labels = labels
    else:
      # Label is downsampled to the same size as logits.
      scaled_labels = tf.image.resize_nearest_neighbor(
            labels,
            preprocess_utils.resolve_shape(logits, 4)[1:3],
            align_corners=True)
      scaled_logits = logits
    assert scaled_labels.get_shape()[1:3] == scaled_logits.get_shape()[1:3], 'The potentially reshaped logits and labels should match in shapes!'
    # assert scaled_labels.dtype == scaled_logits.dtype, 'The potentially reshaped logits and labels should match in types!'
    return scaled_logits, scaled_labels

def euler_angles_to_quaternions(angle):
    """Convert euler angels to quaternions.
    Input:
    angle: [roll, pitch, yaw]
    """
    angle = tf.reshape(angle, [-1, 3])
    roll = tf.gather(angle, [0], axis=1)
    pitch = tf.gather(angle,[1], axis=1)
    yaw = tf.gather(angle, [2], axis=1)
    q = tf.zeros([tf.shape(angle)[0], 4])

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
    q = tf.concat([q0, q1, q2, q3], axis=1)
    return q

def smooth_l1_loss(predictions, labels, weights, name='', loss_collection=tf.GraphKeys.LOSSES):
    loss_sum = tf.losses.huber_loss(labels, predictions, weights=weights, delta=1.0, scope='loss_l1_reg_'+name, loss_collection=loss_collection)
    # loss_sum = tf.losses.absolute_difference(
    #         labels,
    #         predictions)
    # return loss_sum / (tf.reduce_sum(tf.to_float(masks))+1.)
    return loss_sum

def add_my_pose_loss_cars(prob_logits, labels, masks_float, balance_rot=1., balance_trans=1., upsample_logits=True, name=None, loss_collection=None):
    """ Loss for discrete pose from Peng Wang (http://icode.baidu.com/repos/baidu/personal-code/video_seg_transfer/blob/with_db:Networks/mx_losses.py)
    prob_logits, labels: [car_num, D]"""

    def slice_pose(pose_in):
        rot = tf.gather(pose_in, [0, 1, 2, 3], axis=1)
        trans = tf.gather(pose_in, [4, 5, 6], axis=1)
        return rot, trans

    count_valid = tf.reduce_sum(masks_float)+1e-10

    rot, trans = slice_pose(prob_logits)
    rot_gt, trans_gt = slice_pose(labels)

    trans_loss = smooth_l1_loss(trans, trans_gt,
            masks_float, '', loss_collection=None) * balance_trans
    trans_loss = tf.identity(trans_loss, name=name+'_trans')
    tf.losses.add_loss(trans_loss, loss_collection=loss_collection)

    trans_metric = tf.concat([tf.gather(trans, [0, 1], axis=1), 1./tf.gather(trans, [2], axis=1)], axis=1)
    trans_gt_metric = tf.concat([tf.gather(trans_gt, [0, 1], axis=1), 1./tf.gather(trans_gt, [2], axis=1)], axis=1)
    trans_diff_metric = tf.multiply(trans_metric - trans_gt_metric, masks_float)
    trans_error_cars = tf.reduce_sum(tf.square(trans_diff_metric), axis=1, keepdims=True)
    trans_loss_metric_loss = tf.reduce_sum(tf.sqrt(trans_error_cars)) / count_valid
    trans_loss_metric_loss = tf.identity(trans_loss_metric_loss, name=name+'_trans_metric')

    rot_q_loss = tf.reduce_sum(tf.reduce_sum(tf.square(tf.multiply(masks_float, rot - rot_gt)), axis=1) / 2.0) / count_valid
    rot_q_loss = rot_q_loss * balance_rot
    tf.losses.add_loss(tf.identity(rot_q_loss, name=name+'_rot_quat'), loss_collection=loss_collection)
    total_loss = rot_q_loss + trans_loss

    rot_q_unit = tf.nn.l2_normalize(rot, axis=1)
    rot_q_gt_unit = tf.nn.l2_normalize(rot_gt, axis=1)
    # [1/2 Peng] rotation matric following https://github.com/ApolloScapeAuto/dataset-api/blob/master/self_localization/eval_pose.py#L122a
    rot_q_error_cars = tf.acos(tf.abs(1. - tf.reduce_sum(tf.square(rot_q_unit - rot_q_gt_unit) / 2., axis=1, keepdims=True))) * 2 * 180 / np.pi
    dis_rot_metric_loss = tf.reduce_sum(tf.multiply(masks_float, rot_q_error_cars)) / count_valid # per-car angle error
    dis_rot_metric_loss = tf.identity(dis_rot_metric_loss, name=name+'_rot_quat_metric')

    total_loss = tf.identity(total_loss, name=name)
    return total_loss, prob_logits, rot_q_error_cars, trans_error_cars

# def add_my_pose_loss(prob_logits, labels, masks, balance_rot=1., balance_trans=1., upsample_logits=True, name=None, loss_collection=None):
#     """ Loss for discrete pose from Peng Wang (http://icode.baidu.com/repos/baidu/personal-code/video_seg_transfer/blob/with_db:Networks/mx_losses.py)"""

#     scaled_logits, scaled_labels = scale_logits_to_labels(prob_logits, labels, upsample_logits)
#     masks_expanded = tf.tile(masks, [1, 1, 1, tf.shape(labels)[3]])
#     scaled_logits_masked = tf.where(masks_expanded, scaled_logits, tf.zeros_like(scaled_logits))
#     scaled_labels_masked = tf.where(masks_expanded, scaled_labels, tf.zeros_like(scaled_labels))
#     count_valid = tf.reduce_sum(tf.to_float(masks))+1e-6

#     def slice_pose(pose_in):
#         rot = tf.gather(pose_in, [0, 1, 2, 3], axis=3)
#         trans = tf.gather(pose_in, [4, 5, 6], axis=3)
#         return rot, trans

#     rot, trans = slice_pose(scaled_logits_masked)
#     rot_gt, trans_gt = slice_pose(scaled_labels_masked)

#     trans_loss = smooth_l1_loss(trans, trans_gt,
#             tf.to_float(tf.tile(masks, [1, 1, 1, tf.shape(trans)[3]])), '', loss_collection=None) * balance_trans
#     trans_loss = tf.identity(trans_loss, name=name+'_trans')
#     tf.losses.add_loss(trans_loss, loss_collection=loss_collection)

#     trans_metric = tf.concat([tf.gather(trans, [0, 1], axis=3), 1./tf.gather(trans, [2], axis=3)], axis=3)
#     trans_metric = tf.where(tf.tile(masks, [1, 1, 1, tf.shape(trans_metric)[3]]), trans_metric, tf.zeros_like(trans_metric))
#     trans_gt_metric = tf.concat([tf.gather(trans_gt, [0, 1], axis=3), 1./tf.gather(trans_gt, [2], axis=3)], axis=3)
#     trans_gt_metric = tf.where(tf.tile(masks, [1, 1, 1, tf.shape(trans_gt_metric)[3]]), trans_gt_metric, tf.zeros_like(trans_gt_metric))
#     trans_diff_metric = trans_metric - trans_gt_metric
#     trans_error_map = tf.sqrt(tf.reduce_sum(tf.square(trans_diff_metric), axis=3, keepdims=True))
#     trans_loss_metric_loss = tf.nn.l2_loss(trans_diff_metric) / count_valid
#     trans_loss_metric_loss = tf.identity(trans_loss_metric_loss, name=name+'_trans_metric')

#     # rot_flatten = tf.reshape(rot, [-1, 3])
#     # rot_gt_flatten = tf.reshape(rot_gt, [-1, 3])
#     # rot_q_flatten = tf.nn.l2_normalize(euler_angles_to_quaternions(rot_flatten), axis=1)
#     # rot_gt_q_flatten = tf.nn.l2_normalize(euler_angles_to_quaternions(rot_gt_flatten), axis=1)
#     # rot_loss_quat = tf.nn.l2_loss(rot_gt_q_flatten - rot_q_flatten) / tf.to_float(tf.shape(rot_gt_flatten)[0])
#     # rot_loss_quat = tf.identity(rot_loss_quat, name=name+'_rot_quat')

#     # [1/3] Train with l1 loss, show with quat
#     # rot_loss = smooth_l1_loss(rot, rot_gt)
#     # total_loss = rot_loss*balance + trans_loss

#     # [2/3] Train with quat loss
#     # tf.losses.add_loss(rot_loss_quat, loss_collection=tf.GraphKeys.LOSSES)
#     # rot_loss = smooth_l1_loss(rot, rot_gt, loss_collection=None)
#     # total_loss = rot_loss_quat + trans_loss

#     # [3/3] Train with reg to quar loss
#     rot_q_loss = tf.reduce_sum(tf.reduce_sum(tf.square(rot - rot_gt), axis=3) / 2.0)
#     rot_q_loss = rot_q_loss / count_valid * balance_rot
#     tf.losses.add_loss(tf.identity(rot_q_loss, name=name+'_rot_quat'), loss_collection=loss_collection)
#     total_loss = rot_q_loss + trans_loss

#     rot_q_unit = tf.nn.l2_normalize(rot, axis=3)
#     rot_q_gt_unit = tf.nn.l2_normalize(rot_gt, axis=3)
#     # [1/2 Peng] rotation matric following https://github.com/ApolloScapeAuto/dataset-api/blob/master/self_localization/eval_pose.py#L122a
#     rot_q_error_map = tf.acos(tf.abs(1. - tf.reduce_sum(tf.square(rot_q_unit - rot_q_gt_unit) / 2., axis=3, keepdims=True))) * 2 * 180 / np.pi
#     # [2/2 posenet] https://github.com/kentsommer/tensorflow-posenet/blob/master/test.py#L154 OR https://chrischoy.github.io/research/measuring-rotation/ OR https://math.stackexchange.com/questions/90081/quaternion-distance
#     # rot_q_diff_metric = tf.acos(tf.abs(tf.reduce_sum(rot_q_unit * rot_q_gt_unit, axis=3, keep_dims=True)))

#     rot_q_error_map = tf.where(masks, rot_q_error_map, tf.zeros_like(rot_q_error_map))
#     dis_rot_metric_loss = tf.reduce_sum(rot_q_error_map) / count_valid # per-pixel angle error
#     dis_rot_metric_loss = tf.identity(dis_rot_metric_loss, name=name+'_rot_quat_metric')

#     total_loss = tf.identity(total_loss, name=name)
#     return total_loss, scaled_logits, rot_q_error_map, trans_error_map

def logits_cls_to_logits_probReg(logits, bin_vals):
    prob = tf.contrib.layers.softmax(logits)
    # bin_vals_expand = tf.expand_dims(tf.expand_dims(bin_vals, 0), 0)
    bin_vals_expand = tf.tile(bin_vals, [tf.shape(prob)[0], 1])
    # print prob.get_shape(), bin_vals_expand.get_shape()
    prob = tf.multiply(prob, bin_vals_expand)
    prob_logits = tf.reduce_sum(prob, axis=1, keepdims=True)
    return prob_logits

def scale_for_l1_loss(logits, labels, masks, upsample_logits):
    scaled_logits, scaled_labels = scale_logits_to_labels(logits, labels, upsample_logits)
    masks_expanded = tf.tile(masks, [1, 1, 1, tf.shape(scaled_logits)[3]])
    scaled_logits_masked = tf.where(masks_expanded, scaled_logits, tf.zeros_like(scaled_logits))
    return scaled_logits_masked

def add_l1_regression_loss_cars(logits,
        labels,
        masks_float,
        balance=1.,
        upsample_logits=True,
        name='',
        loss_collection=None):
    """Adds softmax cross entropy loss for logits of each scale.

    Args:
      scales_to_logits: A map from logits names for different scales to logits.
        The logits have shape [batch, logits_height, logits_width, num_classes]. # {'merged_logits': <tf.Tensor 'regression:0' shape=(4, 49, 49, 6) dtype=float32>}
      labels: Groundtruth labels with shape [batch, image_height, image_width, 6].
      num_classes: Integer, ground truth regression lebels dimension.
      ignore_label: Integer, label to ignore.
      loss_weight: Float, loss weight.
      upsample_logits: Boolean, upsample logits or not.
      scope: String, the scope for the loss.

    Raises:
      ValueError: Label or logits is None.
    """
    loss = smooth_l1_loss(logits, labels,
            masks_float,'', loss_collection=None) * balance
    loss = tf.identity(loss, name=name)
    if loss_collection!=None:
        tf.losses.add_loss(loss, loss_collection=tf.GraphKeys.LOSSES)
    return loss, logits

# def add_l1_regression_loss(logits,
#         labels,
#         masks,
#         balance=1.,
#         upsample_logits=True,
#         name='',
#         loss_collection=None):
#     """Adds softmax cross entropy loss for logits of each scale.

#     Args:
#       scales_to_logits: A map from logits names for different scales to logits.
#         The logits have shape [batch, logits_height, logits_width, num_classes]. # {'merged_logits': <tf.Tensor 'regression:0' shape=(4, 49, 49, 6) dtype=float32>}
#       labels: Groundtruth labels with shape [batch, image_height, image_width, 6].
#       num_classes: Integer, ground truth regression lebels dimension.
#       ignore_label: Integer, label to ignore.
#       loss_weight: Float, loss weight.
#       upsample_logits: Boolean, upsample logits or not.
#       scope: String, the scope for the loss.

#     Raises:
#       ValueError: Label or logits is None.
#     """
#     scaled_logits, scaled_labels = scale_logits_to_labels(logits, labels, upsample_logits)
#     masks_expanded = tf.tile(masks, [1, 1, 1, tf.shape(scaled_labels)[3]])
#     scaled_logits_masked = tf.where(masks_expanded, scaled_logits, tf.zeros_like(scaled_logits))
#     scaled_labels_masked = tf.where(masks_expanded, scaled_labels, tf.zeros_like(scaled_labels))
#     # print scaled_logits.get_shape(), scaled_labels.get_shape(), masks_expanded.get_shape()

#     loss = smooth_l1_loss(scaled_logits_masked, scaled_labels_masked,
#             tf.to_float(masks_expanded),'', loss_collection=None) * balance
#     # loss = tf.losses.mean_squared_error(
#     #         scaled_labels_masked,
#     #         scaled_logits_masked,
#     #         weights=masks_expanded,
#     #         loss_collection=loss_collection) * balance
#     loss = tf.identity(loss, name=name)
#     if loss_collection!=None:
#         tf.losses.add_loss(loss, loss_collection=tf.GraphKeys.LOSSES)
#     return loss, scaled_logits_masked



def get_model_init_fn(restore_logdir,
                      tf_initial_checkpoint,
                      restore_logged,
                      initialize_last_layer,
                      last_layers,
                      ignore_missing_vars=False):
  """Gets the function initializing model variables from a checkpoint.

  Args:
    train_logdir: Log directory for training.
    tf_initial_checkpoint: TensorFlow checkpoint for initialization.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    ignore_missing_vars: Ignore missing variables in the checkpoint.

  Returns:
    Initialization function.
  """
  print restore_logdir
  if tf_initial_checkpoint is None:
    tf.logging.info('==== Not initializing the model from the initial checkpoint (not given).')
  else:
    exclude_list = ['global_step']
    # return None

  if tf.train.latest_checkpoint(restore_logdir) and restore_logged:
    tf_initial_checkpoint = tf.train.latest_checkpoint(restore_logdir)
    tf.logging.info('==== Ignoring initialization; restoring from logged checkpoint: %s'%tf_initial_checkpoint)
    exclude_list = []

  tf.logging.info('==== Initializing model from path: %s', tf_initial_checkpoint)

  # Variables that will not be restored.
  if not initialize_last_layer and last_layers!=None:
    exclude_list.extend(last_layers)

  variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)

  return slim.assign_from_checkpoint_fn(
      tf_initial_checkpoint,
      variables_to_restore,
      ignore_missing_vars=ignore_missing_vars)


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
  """Gets the gradient multipliers.

  The gradient multipliers will adjust the learning rates for model
  variables. For the task of semantic segmentation, the models are
  usually fine-tuned from the models trained on the task of image
  classification. To fine-tune the models, we usually set larger (e.g.,
  10 times larger) learning rate for the parameters of last layer.

  Args:
    last_layers: Scopes of last layers.
    last_layer_gradient_multiplier: The gradient multiplier for last layers.

  Returns:
    The gradient multiplier map with variables as key, and multipliers as value.
  """
  gradient_multipliers = {}

  for var in slim.get_model_variables():
    # Double the learning rate for biases.
    if 'biases' in var.op.name:
      gradient_multipliers[var.op.name] = 2.

    # Use larger learning rate for last layer variables.
    for layer in last_layers:
      if layer in var.op.name and 'biases' in var.op.name:
        gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
        break
      elif layer in var.op.name:
        gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
        break

  return gradient_multipliers

def filter_gradients(last_layers, grads_and_vars):
    filtered_grads_and_vars = []
    for grad_and_var in grads_and_vars:
        var_name = grad_and_var[1].op.name
        for layer in last_layers:
            if layer in var_name:
                filtered_grads_and_vars.append(grad_and_var)
    return filtered_grads_and_vars

def get_model_learning_rate(
    learning_policy, base_learning_rate, learning_rate_decay_step,
    learning_rate_decay_factor, training_number_of_steps, learning_power,
    slow_start_step, slow_start_learning_rate):
  """Gets model's learning rate.

  Computes the model's learning rate for different learning policy.
  Right now, only "step" and "poly" are supported.
  (1) The learning policy for "step" is computed as follows:
    current_learning_rate = base_learning_rate *
      learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
  See tf.train.exponential_decay for details.
  (2) The learning policy for "poly" is computed as follows:
    current_learning_rate = base_learning_rate *
      (1 - global_step / training_number_of_steps) ^ learning_power

  Args:
    learning_policy: Learning rate policy for training.
    base_learning_rate: The base learning rate for model training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    training_number_of_steps: Number of steps for training.
    learning_power: Power used for 'poly' learning policy.
    slow_start_step: Training model with small learning rate for the first
      few steps.
    slow_start_learning_rate: The learning rate employed during slow start.

  Returns:
    Learning rate for the specified learning policy.

  Raises:
    ValueError: If learning policy is not recognized.
  """
  global_step = tf.train.get_or_create_global_step()
  if learning_policy == 'step':
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step,
        learning_rate_decay_step,
        learning_rate_decay_factor,
        staircase=True)
  elif learning_policy == 'poly':
    learning_rate = tf.train.polynomial_decay(
        base_learning_rate,
        global_step,
        training_number_of_steps,
        end_learning_rate=0,
        power=learning_power)
  else:
    raise ValueError('Unknown learning policy.')

  # Employ small learning rate at the first few steps for warm start.
  return tf.where(global_step < slow_start_step, slow_start_learning_rate,
                  learning_rate)
