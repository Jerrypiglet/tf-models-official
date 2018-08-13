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
"""Training script for the DeepLab model.

See model.py for more details and usage.
"""

import six
import os
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import regression_dataset
from deeplab.utils import input_generator
from deeplab.utils import train_utils
from deployment import model_deploy
import numpy as np
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=4)

from deeplab.core import preprocess_utils

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue

flags = tf.app.flags

FLAGS = flags.FLAGS

# Settings for multi-GPUs/multi-replicas training.

flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID.')

# Settings for logging.

flags.DEFINE_string('task_name', 'tmp',
                    'Task name; will be appended to FLAGS.train_logdir to log files.')

flags.DEFINE_string('restore_name', None,
                    'Task name to restore; will be appended to FLAGS.train_logdir to log files.')

flags.DEFINE_string('base_logdir', None,
                    'Where the checkpoint and logs are stored (base dir).')

flags.DEFINE_string('train_logdir', None,
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_string('restore_logdir', None,
                    'Where the checkpoint and logs are REstored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_boolean('if_val', False,
                     'If we VALIDATE the model.')

flags.DEFINE_integer('val_interval_steps', 10,
                     'How often, in steps, we VALIDATE the model.')

flags.DEFINE_integer('save_interval_secs', 300,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 30,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as images to summary.')

flags.DEFINE_boolean('if_print_tensors', False,
                     'If we print all the tensors and their names.')

# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 5000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 300000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('train_batch_size', 8,
                     'The number of images in each batch during training.')

# For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
flags.DEFINE_boolean('if_discrete_loss', True,
                     'Use discrete regression + classification loss.')

flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_multi_integer('train_crop_size', [513, 513],
                           'Image crop size [height, width] during training.')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

# Settings for fine-tuning the network.

flags.DEFINE_boolean('restore_logged', False,
                    'Whether to restore the logged checkpoint.')

flags.DEFINE_string('tf_initial_checkpoint', None,
                    'The initial checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_feature_extractor', True,
                     'Fine tune the feature extractors or not.')

flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')

flags.DEFINE_float('min_scale_factor', 1.0,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 1.0,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 0.,
                   'Scale factor step size for data augmentation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.
flags.DEFINE_string('dataset', 'apolloscape',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('val_split', 'val',
                    'Which split of the dataset to be used for validation')

flags.DEFINE_string('dataset_dir', 'deeplab/datasets/apolloscape', 'Where the dataset reside.')


def _build_deeplab(inputs_queue, outputs_to_num_classes, outputs_to_indices, bin_vals, bin_nums, pose_range, output_names, is_training=True, reuse=False):
  """Builds a clone of DeepLab.

  Args:
    inputs_queue: A prefetch queue for images and labels.
    # outputs_to_num_classes: A map from output type to the number of classes.
    #   For example, for the task of semantic segmentation with 21 semantic
    #   classes, we would have outputs_to_num_classes['semantic'] = 21.

  Returns:
    A map of maps from output_type (e.g., semantic prediction) to a
      dictionary of multi-scale logits names to logits. For each output_type,
      the dictionary has keys which correspond to the scales and values which
      correspond to the logits. For example, if `scales` equals [1.0, 1.5],
      then the keys would include 'merged_logits', 'logits_1.00' and
      'logits_1.50'.
  """
  samples = inputs_queue.dequeue()

  if is_training:
      is_training_prefix = ''
  else:
      is_training_prefix = 'val-'

  # Add name to input and label nodes so we can add to summary.
  samples[common.IMAGE] = tf.identity(
      samples[common.IMAGE], name=is_training_prefix+common.IMAGE)
  samples[common.IMAGE_NAME] = tf.identity(
      samples[common.IMAGE_NAME], name=is_training_prefix+common.IMAGE_NAME)
  samples['vis'] = tf.identity(samples['vis'], name=is_training_prefix+'vis')
  samples[common.LABEL] = tf.identity(
      samples[common.LABEL], name=is_training_prefix+common.LABEL)
  samples['seg'] = tf.identity(samples['seg'], name=is_training_prefix+'seg')
  # samples['label_id'] = tf.squeeze(samples['label_id'])
  # samples['label_id'] = tf.identity(samples['label_id'], name=is_training_prefix+'label_id')

  model_options = common.ModelOptions(
      outputs_to_num_classes=outputs_to_num_classes,
      crop_size=FLAGS.train_crop_size,
      atrous_rates=FLAGS.atrous_rates,
      output_stride=FLAGS.output_stride)

  outputs_to_logits = model.single_scale_logits(
      samples[common.IMAGE],
      model_options=model_options,
      weight_decay=FLAGS.weight_decay,
      is_training=is_training,
      fine_tune_batch_norm=FLAGS.fine_tune_batch_norm and is_training,
      fine_tune_feature_extractor=FLAGS.fine_tune_feature_extractor and is_training)
  # print outputs_to_logits, 'outputs_to_scales_to_logits @_build_deeplab @train_apolloscape_instance.py' # {'y': <tf.Tensor 'logits_1/y/BiasAdd:0' shape=(5, 68, 170, 16) dtype=float32>, 'x': <tf.Tensor 'logits/x/BiasAdd:0' shape=(5, 68, 170, 16) dtype=float32>, 'z': <tf.Tensor 'logits_2/z/BiasAdd:0' shape=(5, 68, 170, 64) dtype=float32>}


  # Get regressed logits for 6 outputs
  scaled_logits_list = []
  reg_logits_list = []
  for output in output_names:
      label_slice = tf.gather(samples[common.LABEL], [outputs_to_indices[output]], axis=3)
      if FLAGS.if_discrete_loss:
          print outputs_to_logits[output]
          prob_logits = train_utils.logits_cls_to_logits_prob(
                  outputs_to_logits[output],
                  bin_vals[outputs_to_indices[output]])
          # print output, prob_logits.get_shape()
          reg_logits = prob_logits
      else:
          reg_logits = outputs_to_logits[output]
      reg_logits_list.append(reg_logits)

  # Add loss for regressed digits
  reg_logits_concat = tf.concat(reg_logits_list, axis=3)
  # loss, scaled_logits = train_utils.add_regression_loss(
  #         reg_logits_concat,
  #         samples[common.LABEL],
  #         samples['mask'],
  #         loss_weight=1.0,
  #         upsample_logits=FLAGS.upsample_logits,
  #         name=is_training_prefix + 'loss_all'
  #         )
  loss, scaled_logits = train_utils.add_my_pose_loss(
          reg_logits_concat,
          samples[common.LABEL],
          samples['mask'],
          upsample_logits=FLAGS.upsample_logits,
          name=is_training_prefix + 'loss_all'
          )
  scaled_logits = tf.identity(scaled_logits, name=is_training_prefix+'scaled_logits')
  masks = tf.identity(samples['mask'], name=is_training_prefix+'not_ignore_mask_in_loss')
  count_valid = tf.reduce_sum(tf.to_float(masks))+1e-6

  bin_range = [np.linspace(r[0], r[1], num=b).tolist() for r, b in zip(pose_range, bin_nums)]
  label_id_list = []
  loss_slice_crossentropy_list = []
  for idx_output, output in enumerate(output_names):
    # Get label_id slice
    label_slice = tf.gather(samples[common.LABEL], [idx_output], axis=3)
    bin_vals_output = bin_range[idx_output]
    label_id_slice = tf.round((label_slice - bin_vals_output[0]) / (bin_vals_output[1] - bin_vals_output[0]))
    label_id_slice = tf.clip_by_value(label_id_slice, 0, bin_nums[idx_output]-1)
    label_id_slice = tf.cast(label_id_slice, tf.uint8)
    label_id_list.append(label_id_slice)

    # Add losses for each output names for logging
    scaled_logits_slice = tf.gather(scaled_logits, [idx_output], axis=3)
    scaled_logits_slice_masked = tf.where(masks, scaled_logits_slice, tf.zeros_like(scaled_logits_slice))
    loss_slice_reg = tf.losses.huber_loss(label_slice, scaled_logits_slice_masked, delta=1.0, loss_collection=None) / (tf.reduce_sum(tf.to_float(masks))+1e-6)
    loss_slice_reg = tf.identity(loss_slice_reg, name=is_training_prefix+'loss_reg_'+output)

    # Cross-entropy loss for each output http://icode.baidu.com/repos/baidu/personal-code/video_seg_transfer/blob/with_db:Networks/mx_losses.py (L89)
    scaled_logits_disc_slice, _ = train_utils.scale_logits_to_labels(outputs_to_logits[output], label_slice, True)
    neg_log = -1. * tf.nn.log_softmax(scaled_logits_disc_slice)
    gt_idx = tf.one_hot(tf.squeeze(label_id_slice), depth=bin_nums[idx_output], axis=-1)
    print label_id_slice.get_shape(), gt_idx.get_shape(), neg_log.get_shape(), masks.get_shape()
    loss_slice_crossentropy = tf.reduce_sum(tf.multiply(gt_idx, neg_log), axis=3, keep_dims=True)
    loss_slice_crossentropy = tf.where(masks, loss_slice_crossentropy, tf.zeros_like(loss_slice_crossentropy))
    loss_slice_crossentropy= tf.reduce_sum(loss_slice_crossentropy) / count_valid * 1e-1
    loss_slice_crossentropy = tf.identity(loss_slice_crossentropy, name=is_training_prefix+'loss_cls_'+output)
    tf.losses.add_loss(loss_slice_crossentropy, loss_collection=tf.GraphKeys.LOSSES)
    loss_slice_crossentropy_list.append(loss_slice_crossentropy)
  loss_crossentropy = tf.identity(tf.add_n(loss_slice_crossentropy_list), name=is_training_prefix+'loss_cls_ALL')
  label_id = tf.concat(label_id_list, axis=3)
  label_id_masked = tf.where(tf.tile(masks, [1, 1, 1, len(bin_nums)]), label_id, tf.zeros_like(label_id))
  label_id_masked = tf.identity(label_id_masked, name=is_training_prefix+'label_id')

def main(unused_argv):
  FLAGS.train_logdir = FLAGS.base_logdir + '/' + FLAGS.task_name
  if FLAGS.restore_name == None:
      FLAGS.restore_logdir = FLAGS.train_logdir
  else:
      FLAGS.restore_logdir = FLAGS.base_logdir + '/' + FLAGS.restore_name

  tf.logging.set_verbosity(tf.logging.INFO)
  # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
  config = model_deploy.DeploymentConfig(
      num_clones=FLAGS.num_clones,
      clone_on_cpu=FLAGS.clone_on_cpu,
      replica_id=FLAGS.task,
      num_replicas=FLAGS.num_replicas,
      num_ps_tasks=FLAGS.num_ps_tasks) # /device:CPU:0

  # Split the batch across GPUs.
  assert FLAGS.train_batch_size % config.num_clones == 0, (
      'Training batch size not divisble by number of clones (GPUs).')
  clone_batch_size = FLAGS.train_batch_size // config.num_clones

  # Get dataset-dependent information.
  dataset = regression_dataset.get_dataset(
      FLAGS.dataset, FLAGS.train_split, dataset_dir=FLAGS.dataset_dir)
  dataset_val = regression_dataset.get_dataset(
      FLAGS.dataset, FLAGS.val_split, dataset_dir=FLAGS.dataset_dir)
  print '#### The data has size:', dataset.num_samples

  # Get logging dir ready.
  if not(os.path.isdir(FLAGS.train_logdir)):
      tf.gfile.MakeDirs(FLAGS.train_logdir)
  elif len(os.listdir(FLAGS.train_logdir) ) != 0:
      if_delete_all = raw_input('#### The log folder %s exists and non-empty; delete all logs? [y/n] '%FLAGS.train_logdir)
      if if_delete_all == 'y':
          os.system('rm -rf %s/*'%FLAGS.train_logdir)
          print '==== Log folder emptied.'
  tf.logging.info('==== Logging in dir:%s; Training on %s set', FLAGS.train_logdir, FLAGS.train_split)

  with tf.Graph().as_default() as graph:
    with tf.device(config.inputs_device()):
      bin_range = [np.linspace(r[0], r[1], num=b).tolist() for r, b in zip(dataset.pose_range, dataset.bin_nums)]
      outputs_to_num_classes = {}
      outputs_to_indices = {}
      for output, bin_num, idx in zip(dataset.output_names, dataset.bin_nums,range(len(dataset.output_names))):
          if FLAGS.if_discrete_loss:
            outputs_to_num_classes[output] = bin_num
          else:
           outputs_to_num_classes[output] = 1
          outputs_to_indices[output] = idx
      bin_vals = [tf.constant(value=[bin_range[i]], dtype=tf.float32, shape=[1, dataset.bin_nums[i]], name=name) \
                  for i, name in enumerate(dataset.output_names)]
      print outputs_to_num_classes
      # print spaces_to_indices

      samples = input_generator.get(
          dataset,
          FLAGS.train_crop_size,
          clone_batch_size,
          min_resize_value=FLAGS.min_resize_value,
          max_resize_value=FLAGS.max_resize_value,
          resize_factor=FLAGS.resize_factor,
          min_scale_factor=FLAGS.min_scale_factor,
          max_scale_factor=FLAGS.max_scale_factor,
          scale_factor_step_size=FLAGS.scale_factor_step_size,
          dataset_split=FLAGS.train_split,
          is_training=True,
          model_variant=FLAGS.model_variant)
      inputs_queue = prefetch_queue.prefetch_queue(
          samples, capacity=128 * config.num_clones)

      samples_val = input_generator.get(
          dataset_val,
          FLAGS.train_crop_size,
          clone_batch_size,
          min_resize_value=FLAGS.min_resize_value,
          max_resize_value=FLAGS.max_resize_value,
          resize_factor=FLAGS.resize_factor,
          min_scale_factor=FLAGS.min_scale_factor,
          max_scale_factor=FLAGS.max_scale_factor,
          scale_factor_step_size=FLAGS.scale_factor_step_size,
          dataset_split=FLAGS.val_split,
          is_training=False,
          model_variant=FLAGS.model_variant)
      inputs_queue_val = prefetch_queue.prefetch_queue(
          samples_val, capacity=128)

    # Create the global step on the device storing the variables.
    with tf.device(config.variables_device()):
      global_step = tf.train.get_or_create_global_step()

      # Define the model and create clones.
      model_fn = _build_deeplab
      model_args = (inputs_queue, outputs_to_num_classes, outputs_to_indices, bin_vals, dataset.bin_nums, dataset.pose_range, dataset.output_names, True, False)
      clones = model_deploy.create_clones(config, model_fn, args=model_args)

      # Gather update_ops from the first clone. These contain, for example,
      # the updates for the batch_norm variables created by model_fn.
      first_clone_scope = config.clone_scope(0) # clone_0
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    with tf.device('/device:GPU:3'):
        if FLAGS.if_val:
          ## Construct the validation graph; takes one GPU.
          _build_deeplab(inputs_queue_val, outputs_to_num_classes, outputs_to_indices, bin_vals, dataset.bin_nums, dataset.pose_range, dataset.output_names, is_training=False, reuse=True)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for images, labels, semantic predictions
    if FLAGS.save_summaries_images:
      if FLAGS.num_clones > 1:
          pattern_train = first_clone_scope + '/%s:0'
      else:
          pattern_train = '%s:0'
      pattern_val = 'val-%s:0'
      pattern = pattern_val if FLAGS.if_val else pattern_train

      summary_mask = graph.get_tensor_by_name(pattern%'not_ignore_mask_in_loss')
      summary_mask = tf.reshape(summary_mask, [-1, FLAGS.train_crop_size[0], FLAGS.train_crop_size[1], 1])
      summary_mask_float = tf.to_float(summary_mask)
      summaries.add(tf.summary.image('gth/%s' % 'not_ignore_mask', tf.gather(tf.cast(summary_mask_float*255., tf.uint8), [0, 1, 2])))

      summary_image = graph.get_tensor_by_name(pattern%common.IMAGE)
      summaries.add(tf.summary.image('gth/%s' % common.IMAGE, tf.gather(summary_image, [0, 1, 2])))

      summary_image_name = graph.get_tensor_by_name(pattern%common.IMAGE_NAME)
      summaries.add(tf.summary.text('gth/%s' % common.IMAGE_NAME, tf.gather(summary_image_name, [0, 1, 2])))

      summary_vis = graph.get_tensor_by_name(pattern%'vis')
      summaries.add(tf.summary.image('gth/%s' % 'vis', tf.gather(summary_vis, [0, 1, 2])))

      def scale_to_255(tensor, pixel_scaling=None):
          if pixel_scaling == None:
              offset_to_zero = tf.reduce_min(tensor)
              scale_to_255 = tf.div(255., tf.reduce_max(tensor - offset_to_zero))
          else:
              offset_to_zero, scale_to_255 = pixel_scaling
          summary_tensor_float = tensor - offset_to_zero
          summary_tensor_float = summary_tensor_float * scale_to_255
          summary_tensor_float = tf.clip_by_value(summary_tensor_float, 0., 255.)
          summary_tensor_uint8 = tf.cast(summary_tensor_float, tf.uint8)
          return summary_tensor_uint8, (offset_to_zero, scale_to_255)

      label_outputs = graph.get_tensor_by_name(pattern%common.LABEL)
      label_id_outputs = graph.get_tensor_by_name(pattern%'label_id')
      logit_outputs = graph.get_tensor_by_name(pattern%'scaled_logits')

      for output_idx, output in enumerate(dataset.output_names):
          # # Scale up summary image pixel values for better visualization.
          summary_label_output = tf.gather(label_outputs, [output_idx], axis=3)
          summary_label_output= tf.where(summary_mask, summary_label_output, tf.zeros_like(summary_label_output))
          # pixel_scaling = tf.div(255., tf.reduce_max(tf.where(tf.not_equal(summary_label_output, 255.), summary_label_output, tf.zeros_like(summary_label_output))))
          # summary_label_output_uint8 = tf.cast(summary_label_output * pixel_scaling, tf.uint8)
          summary_label_output_uint8, pixel_scaling = scale_to_255(summary_label_output)
          summaries.add(tf.summary.image('output/%s_label' % output, tf.gather(summary_label_output_uint8, [0, 1, 2])))

          summary_logit_output = tf.gather(logit_outputs, [output_idx], axis=3)
          summary_logit_output = tf.where(summary_mask, summary_logit_output, tf.zeros_like(summary_logit_output))
          summary_logit_output_uint8, _ = scale_to_255(summary_logit_output, pixel_scaling)
          summaries.add(tf.summary.image(
              'output/%s_logit' % output, tf.gather(summary_logit_output_uint8, [0, 1, 2])))

          summary_label_id_output = tf.to_float(tf.gather(label_id_outputs, [output_idx], axis=3))
          summary_label_id_output = tf.where(summary_mask, summary_label_id_output+1, tf.zeros_like(summary_label_id_output))
          summary_label_id_output_uint8, _ = scale_to_255(summary_label_id_output)
          summary_label_id_output_uint8 = tf.identity(summary_label_id_output_uint8, 'tttt'+output)
          summaries.add(tf.summary.image(
              'test/%s_label_id' % output, tf.gather(summary_label_id_output_uint8, [0, 1, 2])))

          summary_diff = tf.abs(tf.to_float(summary_label_output_uint8) - tf.to_float(summary_logit_output_uint8))
          summary_diff = tf.where(summary_mask, summary_diff, tf.zeros_like(summary_diff))
          summaries.add(tf.summary.image('output/%s_ldiff' % output, tf.gather(tf.cast(summary_diff, tf.uint8), [0, 1, 2])))

          summary_loss = graph.get_tensor_by_name((pattern%'loss_reg_').replace(':0', '')+output+':0')
          summaries.add(tf.summary.scalar('slice_loss/'+(pattern%'_loss_reg_').replace(':0', '')+output, summary_loss))

          summary_loss = graph.get_tensor_by_name((pattern%'loss_cls_').replace(':0', '')+output+':0')
          summaries.add(tf.summary.scalar('slice_loss/'+(pattern%'_loss_cls_').replace(':0', '')+output, summary_loss))

      for pattern in [pattern_train, pattern_val] if FLAGS.if_val else [pattern_train]:
          summary_loss = graph.get_tensor_by_name(pattern%'loss_all')
          summaries.add(tf.summary.scalar(('total_loss/'+pattern%'loss_all').replace(':0', ''), summary_loss))

          summary_loss_rot = graph.get_tensor_by_name(pattern%'loss_all_rot_quat_metric')
          summaries.add(tf.summary.scalar(('total_loss/'+pattern%'loss_all_rot_quat_metric').replace(':0', ''), summary_loss_rot))

          summary_loss_rot = graph.get_tensor_by_name(pattern%'loss_all_rot_quat')
          summaries.add(tf.summary.scalar(('total_loss/'+pattern%'loss_all_rot_quat').replace(':0', ''), summary_loss_rot))

          summary_loss_trans = graph.get_tensor_by_name(pattern%'loss_all_trans')
          summaries.add(tf.summary.scalar(('total_loss/'+pattern%'loss_all_trans').replace(':0', ''), summary_loss_trans))

          summary_loss_trans = graph.get_tensor_by_name(pattern%'loss_cls_ALL')
          summaries.add(tf.summary.scalar(('total_loss/'+pattern%'loss_cls_ALL').replace(':0', ''), summary_loss_trans))


    # Build the optimizer based on the device specification.
    with tf.device(config.optimizer_device()):
      learning_rate = train_utils.get_model_learning_rate(
          FLAGS.learning_policy, FLAGS.base_learning_rate,
          FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
          FLAGS.training_number_of_steps, FLAGS.learning_power,
          FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
      # optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps

    with tf.device(config.variables_device()):
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, optimizer)
      print '------ total_loss', total_loss, tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope)
      total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
      summaries.add(tf.summary.scalar('total_loss/train', total_loss))

      # Modify the gradients for biases and last layer variables.
      last_layers = model.get_extra_layer_scopes(
          FLAGS.last_layers_contain_logits_only)
      print '////last layers', last_layers

      # Filter trainable variables for last layers ONLY.
      # grads_and_vars = train_utils.filter_gradients(last_layers, grads_and_vars)

      grad_mult = train_utils.get_model_gradient_multipliers(
          last_layers, FLAGS.last_layer_gradient_multiplier)
      if grad_mult:
        grads_and_vars = slim.learning.multiply_gradients(
            grads_and_vars, grad_mult)

      # Create gradient update op.
      grad_updates = optimizer.apply_gradients(
          grads_and_vars, global_step=global_step)
      update_ops.append(grad_updates)
      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(
        tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries))

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    def train_step_fn(sess, train_op, global_step, train_step_kwargs):
        train_step_fn.step += 1  # or use global_step.eval(session=sess)

        # calc training losses
        loss, should_stop = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)
        # print loss
        # first_clone_test = graph.get_tensor_by_name(
        #         ('%s/%s:0' % (first_clone_scope, 'label_id')).strip('/'))
        # first_clone_test2 = graph.get_tensor_by_name(
        #         ('%s/%s:0' % (first_clone_scope, common.LABEL)).strip('/'))
        #         # 'ttttrow:0')
        # test, test2 = sess.run([first_clone_test, first_clone_test2])
        # test_out = test[:, :, :, 3]
        # test_out2 = test2[:, :, :, 3]
        # # print test_out
        # print test_out.shape, np.max(test_out), np.min(test_out), np.mean(test_out), np.median(test_out), test_out.dtype
        # print test_out2.shape, np.max(test_out2), np.min(test_out2), np.mean(test_out2), np.median(test_out2), test_out2.dtype
        print 'loss: ', loss
        # first_clone_test = graph.get_tensor_by_name(
        #         ('%s/%s:0' % (first_clone_scope, 'loss_all')).strip('/'))
        # test = sess.run(first_clone_test)
        # print test
        should_stop = 0

        if FLAGS.if_val and train_step_fn.step % FLAGS.val_interval_steps == 0:
            first_clone_test = graph.get_tensor_by_name('val-loss_all:0')
            test = sess.run(first_clone_test)
            print '-- Validating... Loss: %.4f'%test
            first_clone_test = graph.get_tensor_by_name(
                    ('%s/%s:0' % (first_clone_scope, 'scaled_logits')).strip('/'))
            first_clone_test2 = graph.get_tensor_by_name(
                    ('%s/%s:0' % (first_clone_scope, common.LABEL)).strip('/'))
                    # 'ttttrow:0')
            test, test2 = sess.run([first_clone_test, first_clone_test2])
            test_out = test[:, :, :, 3]
            test_out = test_out[test_out!=0]
            test_out2 = test2[:, :, :, 3]
            test_out2 = test_out2[test_out2!=0]
            print test_out.shape, np.max(test_out), np.min(test_out), np.mean(test_out), np.median(test_out), test_out.dtype
            print test_out2.shape, np.max(test_out2), np.min(test_out2), np.mean(test_out2), np.median(test_out2), test_out2.dtype

        # first_clone_label = graph.get_tensor_by_name(
        #         ('%s/%s:0' % (first_clone_scope, common.LABEL)).strip('/')) # clone_0/val-loss:0
        # # first_clone_pose_dict = graph.get_tensor_by_name(
        # #         ('%s/%s:0' % (first_clone_scope, 'pose_dict')).strip('/'))
        # first_clone_logit = graph.get_tensor_by_name(
        #         ('%s/%s:0' % (first_clone_scope, 'scaled_regression')).strip('/'))
        # not_ignore_mask = graph.get_tensor_by_name(
        #         ('%s/%s:0' % (first_clone_scope, 'not_ignore_mask_in_loss')).strip('/'))
        # label, logits, mask = sess.run([first_clone_label, first_clone_logit, not_ignore_mask])
        # mask = np.reshape(mask, (-1, FLAGS.train_crop_size[0], FLAGS.train_crop_size[1], dataset.num_classes))

        # print '... shapes, types, loss', label.shape, label.dtype, logits.shape, logits.dtype, loss
        # print 'mask', mask.shape, np.mean(mask)
        # logits[mask==0.] = 0.
        # print 'logits', logits.shape, np.max(logits), np.min(logits), np.mean(logits), logits.dtype
        # for idx in range(6):
        #     print idx, np.max(label[:, :, :, idx]), np.min(label[:, :, :, idx])
        # label = label[:, :, :, 5]
        # print 'label', label.shape, np.max(label), np.min(label), np.mean(label), label.dtype
        # print pose_dict, pose_dict.shape
        # # print 'training....... logits stats: ', np.max(logits), np.min(logits), np.mean(logits)
        # # label_one_piece = label[0, :, :, 0]
        # # print 'training....... label stats', np.max(label_one_piece), np.min(label_one_piece), np.sum(label_one_piece[label_one_piece!=255.])
        return [loss, should_stop]
    train_step_fn.step = 0


    # trainables = [v.name for v in tf.trainable_variables()]
    # alls =[v.name for v in tf.all_variables()]
    # print '----- Trainables %d: '%len(trainables), trainables
    # print '----- All %d: '%len(alls), alls[:10]
    # print '===== ', len(list(set(trainables) - set(alls)))
    # print '===== ', len(list(set(alls) - set(trainables)))

    if FLAGS.if_print_tensors:
        for op in tf.get_default_graph().get_operations():
            print str(op.name)

    # Start the training.
    slim.learning.train(
        train_tensor,
        train_step_fn=train_step_fn,
        logdir=FLAGS.train_logdir,
        log_every_n_steps=FLAGS.log_steps,
        master=FLAGS.master,
        number_of_steps=FLAGS.training_number_of_steps,
        is_chief=(FLAGS.task == 0),
        session_config=session_config,
        startup_delay_steps=startup_delay_steps,
        init_fn=train_utils.get_model_init_fn(
            FLAGS.restore_logdir,
            FLAGS.tf_initial_checkpoint,
            FLAGS.restore_logged,
            FLAGS.initialize_last_layer,
            last_layers,
            ignore_missing_vars=True),
        summary_op=summary_op,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
  flags.mark_flag_as_required('base_logdir')
  flags.mark_flag_as_required('tf_initial_checkpoint')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
