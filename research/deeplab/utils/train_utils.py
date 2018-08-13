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

def scale_logits_to_labels(logits, labels, upsample_logits):
    """ Scaled logits and labels to the same scale."""
    if labels is None:
      raise ValueError('No label for softmax cross entropy loss.')
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
    assert scaled_labels.get_shape()[:3] == scaled_logits.get_shape()[:3], 'The potentially reshaped logits and labels should match in shapes!'
    assert scaled_labels.dtype == scaled_logits.dtype, 'The potentially reshaped logits and labels should match in types!'
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

def smooth_l1_loss(predictions, labels, name='', loss_collection=tf.GraphKeys.LOSSES):
    loss_sum = tf.losses.huber_loss(labels, predictions, delta=1.0, scope='collection_loss_reg_'+name, loss_collection=loss_collection)
    # loss_sum = tf.losses.absolute_difference(
    #         labels,
    #         predictions)
    # return loss_sum / (tf.reduce_sum(tf.to_float(masks))+1.)
    return loss_sum

def add_my_pose_loss(prob_logits, labels, masks, upsample_logits, name=None, loss_type='rel_trans'):
    """ Loss for discrete pose from Peng Wang (http://icode.baidu.com/repos/baidu/personal-code/video_seg_transfer/blob/with_db:Networks/mx_losses.py)"""

    scaled_logits, scaled_labels = scale_logits_to_labels(prob_logits, labels, upsample_logits)
    masks_expanded = tf.tile(masks, [1, 1, 1, tf.shape(labels)[3]])
    scaled_logits_masked = tf.where(masks_expanded, scaled_logits, tf.zeros_like(scaled_logits))
    scaled_labels_masked = tf.where(masks_expanded, scaled_labels, tf.zeros_like(scaled_labels))
    count_valid = tf.reduce_sum(tf.to_float(masks))+1e-6

    def slice_pose(pose_in):
        rot = tf.gather(pose_in, [0, 1, 2, 3], axis=3)
        trans = tf.gather(pose_in, [4, 5, 6], axis=3)
        return rot, trans

    rot, trans = slice_pose(scaled_logits_masked)
    rot_gt, trans_gt = slice_pose(scaled_labels_masked)

    trans_loss = smooth_l1_loss(trans, trans_gt, '', loss_collection=None) / count_valid * 1e5
    trans_loss = tf.identity(trans_loss, name=name+'_trans')
    tf.losses.add_loss(trans_loss, loss_collection=tf.GraphKeys.LOSSES)

    trans_metric = tf.concat([tf.gather(trans, [0, 1], axis=3), 1./tf.gather(trans, [2], axis=3)], axis=3)
    trans_metric = tf.where(tf.tile(masks, [1, 1, 1, tf.shape(trans_metric)[3]]), trans_metric, tf.zeros_like(trans_metric))
    trans_gt_metric = tf.concat([tf.gather(trans_gt, [0, 1], axis=3), 1./tf.gather(trans_gt, [2], axis=3)], axis=3)
    trans_gt_metric = tf.where(tf.tile(masks, [1, 1, 1, tf.shape(trans_gt_metric)[3]]), trans_gt_metric, tf.zeros_like(trans_gt_metric))
    trans_loss_metric = tf.nn.l2_loss(trans_metric - trans_gt_metric) / count_valid
    trans_loss_metric = tf.identity(trans_loss_metric, name=name+'_trans_metric')

    # rot_flatten = tf.reshape(rot, [-1, 3])
    # rot_gt_flatten = tf.reshape(rot_gt, [-1, 3])
    # rot_q_flatten = tf.nn.l2_normalize(euler_angles_to_quaternions(rot_flatten), axis=1)
    # rot_gt_q_flatten = tf.nn.l2_normalize(euler_angles_to_quaternions(rot_gt_flatten), axis=1)
    # rot_loss_quat = tf.nn.l2_loss(rot_gt_q_flatten - rot_q_flatten) / tf.to_float(tf.shape(rot_gt_flatten)[0])
    # rot_loss_quat = tf.identity(rot_loss_quat, name=name+'_rot_quat')

    # [1/2] Train with l1 loss, show with quat
    # rot_loss = smooth_l1_loss(rot, rot_gt)
    # total_loss = rot_loss*balance + trans_loss

    # [2/2] Train with quat loss
    # tf.losses.add_loss(rot_loss_quat, loss_collection=tf.GraphKeys.LOSSES)
    # rot_loss = smooth_l1_loss(rot, rot_gt, loss_collection=None)
    # total_loss = rot_loss_quat + trans_loss

    # [3/3] Train with reg to quar loss
    balance = 10.
    rot_q_diff = tf.reduce_sum(tf.reduce_sum(tf.square(rot - rot_gt), axis=3) / 2.0)
    rot_q_diff = rot_q_diff / count_valid * balance
    tf.losses.add_loss(tf.identity(rot_q_diff, name=name+'_rot_quat'), loss_collection=tf.GraphKeys.LOSSES)
    total_loss = rot_q_diff * balance + trans_loss

    # [???] rotation matric following https://github.com/ApolloScapeAuto/dataset-api/blob/master/self_localization/eval_pose.py#L122a
    # rot_q_diff = tf.abs(1. - tf.reduce_sum(tf.square(rot_q_flatten - rot_gt_q_flatten), axis=1) / 2.0)
    # rot_q_diff_metric = tf.abs(1. - tf.reduce_sum(tf.square(tf.nn.l2_normalize(rot, axis=1) - tf.nn.l2_normalize(rot_gt, axis=1)), axis=1) / 2.0)

    # [posenet] https://github.com/kentsommer/tensorflow-posenet/blob/master/test.py#L154 OR https://chrischoy.github.io/research/measuring-rotation/ OR https://math.stackexchange.com/questions/90081/quaternion-distance
    rot_q_unit = tf.nn.l2_normalize(rot, axis=3)
    rot_q_gt_unit = tf.nn.l2_normalize(rot_gt, axis=3)
    rot_q_diff_metric = tf.acos(tf.abs(tf.reduce_sum(rot_q_unit * rot_q_gt_unit, axis=3, keep_dims=True)))
    rot_q_diff_metric = tf.where(masks, rot_q_diff_metric, tf.zeros_like(rot_q_diff_metric))
    dis_rot_metric = tf.reduce_sum(2 * rot_q_diff_metric * 180 / np.pi) / count_valid # per-pixel angle error
    dis_rot_metric = tf.identity(dis_rot_metric, name=name+'_rot_quat_metric')

    total_loss = tf.identity(total_loss, name=name)
    return total_loss, scaled_logits

def logits_cls_to_logits_prob(logits, bin_vals):
    prob = tf.contrib.layers.softmax(logits)
    bin_vals_expand = tf.expand_dims(tf.expand_dims(bin_vals, 0), 0)
    bin_vals_expand = tf.tile(bin_vals_expand, [tf.shape(prob)[0], tf.shape(prob)[1], tf.shape(prob)[2], 1])
    prob = tf.multiply(prob, bin_vals_expand)
    prob_logits = tf.reduce_sum(prob, axis=3, keepdims=True)
    return prob_logits

def add_regression_loss(logits,
        labels,
        masks,
        # num_classes,
        # ignore_label,
        loss_weight=1.0,
        upsample_logits=True,
        name=None):
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
    scaled_logits, scaled_labels = scaled_logits_labels(logits, labels, upsample_logits)
    masks_expanded = tf.tile(masks, [1, 1, 1, tf.shape(labels)[3]])
    print scaled_logits.get_shape(), scaled_labels.get_shape(), masks_expanded.get_shape()

    scaled_labels_flattened = tf.reshape(scaled_labels, shape=[-1])
    not_ignore_masks_flattened = tf.to_float(tf.reshape(masks_expanded, shape=[-1]))
    scaled_logits_flattened = tf.reshape(scaled_logits, shape=[-1])
    # tf.losses.mean_squared_error(
    loss = tf.losses.absolute_difference(
            scaled_labels_flattened,
            scaled_logits_flattened,
            weights=not_ignore_masks_flattened*loss_weight,
            )  / tf.to_float(tf.shape(labels)[0])
    loss = tf.identity(loss, name=name)
    return loss, scaled_logits



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
  if not initialize_last_layer:
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
