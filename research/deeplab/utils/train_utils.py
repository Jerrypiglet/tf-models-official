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
from deeplab.core import preprocess_utils

slim = tf.contrib.slim

def scaled_logits_labels(logits, labels, upsample_logits):
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
    assert scaled_labels.get_shape() == scaled_logits.get_shape(), 'The potentially reshaped logits and labels should match in shapes!'
    assert scaled_labels.dtype == scaled_logits.dtype, 'The potentially reshaped logits and labels should match in types!'
    return scaled_logits, scaled_labels


def my_pose_loss(pose, label, mask, balance=1000, name='pose_loss', loss_type='rel_trans'):
    """ Loss for discrete pose from Peng Wang (http://icode.baidu.com/repos/baidu/personal-code/video_seg_transfer/blob/with_db:Networks/mx_losses.py)"""

    def slice_pose(pose_in):
        rot = tf.gather(pose_in, [0, 1, 2], axis=3)
        trans = tf.gather(pose_in, [3, 4, 5], axis=3)
        return rot, trans

    def smooth_l1_loss(predictions, labels, masks):
        masks_expanded = tf.tile(masks, [1, 1, 1, labels.get_shape()[3]])
        loss_sum = tf.losses.huber_loss(labels, predictions, delta=1.0, weights=tf.to_float(masks_expanded))
        return loss_sum / tf.to_float(tf.shape(labels)[0])

    rot, trans = slice_pose(pose)
    rot_gt, trans_gt = slice_pose(label)

    if loss_type == 'rel_trans':
        depth_gt = tf.gather(trans_gt, [2], axis=3)
        depth_gt = tf.tile(depth_gt, [1, 1, 1, 3])
        inv_depth_gt = tf.where(tf.not_equal(depth_gt, 0.), 1./depth_gt, tf.zeros_like(depth_gt))
        trans_diff = (trans - trans_gt) * inv_depth_gt

    # elif loss_type == 'sig':
    #     trans_diff = (mx.sym.sigmoid(trans / 30) - mx.sym.sigmoid(trans_gt / 30)) * 2

    # else:
    #     trans_diff = trans - trans_gt

    trans_loss = smooth_l1_loss(trans * inv_depth_gt, trans_gt * inv_depth_gt, mask)
    rot_loss = smooth_l1_loss(rot, rot_gt, mask)

    total_loss = rot_loss * balance + trans_loss
    return total_loss


def add_discrete_regression_loss(logits,
        labels,
        masks, # boolean
        bin_vals,
        # num_classes,
        # ignore_label,
        loss_weight=1.0,
        upsample_logits=True,
        scope=None):
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
    # print logits.get_shape(), labels.get_shape(), masks.get_shape(), bin_vals.get_shape() # (5, 68, 170, 16) (5, 272, 680, 1) (5, 272, 680, 1) (1, 16)
    prob = tf.nn.softmax(logits, axis=3)
    bin_vals_expand = tf.expand_dims(tf.expand_dims(bin_vals, 0), 0)
    bin_vals_expand = tf.tile(bin_vals_expand, [prob.get_shape()[0], prob.get_shape()[1], prob.get_shape()[2], 1])
    prob = tf.multiply(prob, bin_vals_expand)
    prob_logits = tf.reduce_sum(prob, axis=3, keepdims=True)

    scaled_logits, scaled_labels = scaled_logits_labels(prob_logits, labels, upsample_logits)

    loss_reg = my_pose_loss(scaled_logits, scaled_labels, masks)

    return loss_reg, scaled_logits

def add_regression_loss(logits,
        labels,
        masks,
        # num_classes,
        # ignore_label,
        loss_weight=1.0,
        upsample_logits=True,
        scope=None):
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
    print logits.get_shape(), labels.get_shape()
    scaled_logits, scaled_labels = scaled_logits_labels(logits, labels)

    scaled_labels_flattened = tf.reshape(scaled_labels, shape=[-1])
    not_ignore_masks_flattened = tf.to_float(tf.reshape(masks, shape=[-1]))
    # not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels_flattened,
    #                                            ignore_label))
    scaled_logits_flattened = tf.reshape(scaled_logits, shape=[-1])
    # tf.losses.mean_squared_error(
    loss = tf.losses.absolute_difference(
            scaled_labels_flattened,
            scaled_logits_flattened,
            weights=not_ignore_masks_flattened*loss_weight,
            scope=scope
            )  / tf.to_float(tf.shape(labels)[0])
    print '+++++++++++', scaled_labels.get_shape(), scaled_logits.get_shape()
    return loss, scaled_logits


# def add_val_regression_loss(logits, labels, masks, loss_weight=1.0, upsample_logits=True, scope=None):
#   """Adds softmax cross entropy loss for logits of each scale.

#   Args:
#     scales_to_logits: A map from logits names for different scales to logits.
#       The logits have shape [batch, logits_height, logits_width, num_classes]. # {'merged_logits': <tf.Tensor 'regression:0' shape=(4, 49, 49, 6) dtype=float32>}
#     labels: Groundtruth labels with shape [batch, image_height, image_width, 6].
#     loss_weight: Float, loss weight.
#     upsample_logits: Boolean, upsample logits or not.
#     scope: String, the scope for the loss.

#   Raises:
#     ValueError: Label or logits is None.
#   """
#   if labels is None:
#       raise ValueError('No label for softmax cross entropy loss.')
#   if upsample_logits:
#       # Label is not downsampled, and instead we upsample logits.
#       print '+++ [Val] Upsample logits!'
#       scaled_logits = tf.image.resize_bilinear(
#               logits,
#               preprocess_utils.resolve_shape(labels, 4)[1:3],
#               align_corners=True)
#       scaled_labels = labels
#   else:
#       # Label is downsampled to the same size as logits.
#       scaled_labels = tf.image.resize_nearest_neighbor(
#               labels,
#               preprocess_utils.resolve_shape(logits, 4)[1:3],
#               align_corners=True)
#       scaled_logits = logits

#   assert scaled_labels.get_shape() == scaled_logits.get_shape(), 'The potentially reshaped logits and labels should match in shapes!'
#   assert scaled_labels.dtype == scaled_logits.dtype, 'The potentially reshaped logits and labels should match in types!'
#   scaled_labels_flattened = tf.reshape(scaled_labels, shape=[-1])
#   not_ignore_masks_flattened = tf.to_float(tf.reshape(masks, shape=[-1]))
#   scaled_logits_flattened = tf.reshape(scaled_logits, shape=[-1])
#   loss = tf.losses.absolute_difference(
#             scaled_labels_flattened,
#             scaled_logits_flattened,
#             weights=not_ignore_masks_flattened*loss_weight,
#             scope=scope
#             )
#   return loss, scaled_logits

# def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
#                                                   labels,
#                                                   num_classes,
#                                                   ignore_label,
#                                                   loss_weight=1.0,
#                                                   upsample_logits=True,
#                                                   scope=None):
#   """Adds softmax cross entropy loss for logits of each scale.

#   Args:
#     scales_to_logits: A map from logits names for different scales to logits.
#       The logits have shape [batch, logits_height, logits_width, num_classes].
#     labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
#     num_classes: Integer, number of target classes.
#     ignore_label: Integer, label to ignore.
#     loss_weight: Float, loss weight.
#     upsample_logits: Boolean, upsample logits or not.
#     scope: String, the scope for the loss.

#   Raises:
#     ValueError: Label or logits is None.
#   """
#   if labels is None:
#     raise ValueError('No label for softmax cross entropy loss.')

#   for scale, logits in six.iteritems(scales_to_logits):
#     loss_scope = None
#     if scope:
#       loss_scope = '%s_%s' % (scope, scale)

#     if upsample_logits:
#       # Label is not downsampled, and instead we upsample logits.
#       logits = tf.image.resize_bilinear(
#           logits,
#           preprocess_utils.resolve_shape(labels, 4)[1:3],
#           align_corners=True)
#       scaled_labels = labels
#     else:
#       # Label is downsampled to the same size as logits.
#       scaled_labels = tf.image.resize_nearest_neighbor(
#           labels,
#           preprocess_utils.resolve_shape(logits, 4)[1:3],
#           align_corners=True)

#     scaled_labels = tf.reshape(scaled_labels, shape=[-1])
#     not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
#                                                ignore_label)) * loss_weight
#     one_hot_labels = slim.one_hot_encoding(
#         scaled_labels, num_classes, on_value=1.0, off_value=0.0)
#     tf.losses.softmax_cross_entropy(
#         one_hot_labels,
#         tf.reshape(logits, shape=[-1, num_classes]),
#         weights=not_ignore_mask,
#         scope=loss_scope)


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
