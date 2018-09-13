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
import warnings
warnings.filterwarnings("ignore")
import six
import os
import shutil
import tensorflow as tf
from tensorflow import logging
import coloredlogs
coloredlogs.install(level='DEBUG')
tf.logging.set_verbosity(tf.logging.DEBUG)
from deeplab import common
# from deeplab import model
from deeplab import model_maskLogits as model
from deeplab.datasets import regression_dataset_mP as regression_dataset
from deeplab.utils import input_generator_mP as input_generator
from deeplab.utils import train_utils_mP as train_utils
from deployment import model_deploy
import numpy as np
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

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

flags.DEFINE_boolean('if_debug', False,
                     'If in debug mode.')

flags.DEFINE_integer('val_interval_steps', 30,
                     'How often, in steps, we VALIDATE the model.')

flags.DEFINE_integer('save_interval_secs', 300,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 30,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as images to summary.')

flags.DEFINE_boolean('if_print_tensors', False,
                     'If we print all the tensors and their names.')

flags.DEFINE_boolean('if_summary_shape_metrics', True,
                     'Save image metrics to summary.')


# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.2,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 1.,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 100000,
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

flags.DEFINE_boolean('if_restore', False,
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

flags.DEFINE_boolean('if_depth', False,
        'True: regression to depth; False: regression to invd.')

# Dataset settings.
flags.DEFINE_string('dataset', 'apolloscape',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('val_split', 'val',
                    'Which split of the dataset to be used for validation')

flags.DEFINE_string('dataset_dir', 'deeplab/datasets/apolloscape', 'Where the dataset reside.')


from build_deeplab_mP import _build_deeplab

def main(unused_argv):
  FLAGS.train_logdir = FLAGS.base_logdir + '/' + FLAGS.task_name
  if FLAGS.restore_name == None:
      FLAGS.restore_logdir = FLAGS.train_logdir
  else:
      FLAGS.restore_logdir = FLAGS.base_logdir + '/' + FLAGS.restore_name

  # Get logging dir ready.
  if not(os.path.isdir(FLAGS.train_logdir)):
      tf.gfile.MakeDirs(FLAGS.train_logdir)
  if len(os.listdir(FLAGS.train_logdir)) != 0:
      if not(FLAGS.if_restore):
          if FLAGS.if_debug:
              shutil.rmtree(FLAGS.train_logdir)
              print '==== Log folder %s emptied: '%FLAGS.train_logdir + 'rm -rf %s/*'%FLAGS.train_logdir
          else:
              if_delete_all = raw_input('#### The log folder %s exists and non-empty; delete all logs? [y/n] '%FLAGS.train_logdir)
              if if_delete_all == 'y':
                  shutil.rmtree(FLAGS.train_logdir)
                  print '==== Log folder %s emptied: '%FLAGS.train_logdir + 'rm -rf %s/*'%FLAGS.train_logdir
      else:
          print '==== Log folder exists; not emptying it because we need to restore from it.'
  tf.logging.info('==== Logging in dir:%s; Training on %s set', FLAGS.train_logdir, FLAGS.train_split)

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
  dataset = regression_dataset.get_dataset(FLAGS,
      FLAGS.dataset, FLAGS.train_split, dataset_dir=FLAGS.dataset_dir)
  dataset_val = regression_dataset.get_dataset(FLAGS,
      FLAGS.dataset, FLAGS.val_split, dataset_dir=FLAGS.dataset_dir)
  print '#### The data has size:', dataset.num_samples, dataset_val.num_samples

  codes = np.load('./deeplab/codes.npy')

  with tf.Graph().as_default() as graph:
    with tf.device(config.inputs_device()):
      codes_max = np.amax(codes, axis=1).reshape((-1, 1))
      codes_min = np.amin(codes, axis=1).reshape((-1, 1))
      shape_range = np.hstack((codes_max + (codes_max - codes_min)/(dataset.SHAPE_BINS-1.), codes_min - (codes_max - codes_min)/(dataset.SHAPE_BINS-1.)))
      bin_range = [np.linspace(r[0], r[1], num=b).tolist() for r, b in zip(np.vstack((dataset.pose_range, shape_range)), dataset.bin_nums)]
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

      model_options = common.ModelOptions(
        outputs_to_num_classes=outputs_to_num_classes,
        crop_size=[dataset.height, dataset.width],
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

      samples = input_generator.get(
          dataset,
          model_options,
          codes,
          clone_batch_size,
          dataset_split=FLAGS.train_split,
          is_training=True,
          model_variant=FLAGS.model_variant)
      inputs_queue = prefetch_queue.prefetch_queue(
          samples, capacity=64 * config.num_clones, dynamic_pad=True)

      samples_val = input_generator.get(
          dataset_val,
          model_options,
          codes,
          clone_batch_size,
          dataset_split=FLAGS.val_split,
          is_training=False,
          model_variant=FLAGS.model_variant)
      inputs_queue_val = prefetch_queue.prefetch_queue(
          samples_val, capacity=64, dynamic_pad=True)

    # Create the global step on the device storing the variables.
    with tf.device(config.variables_device()):
      global_step = tf.train.get_or_create_global_step()

      # Define the model and create clones.
      model_fn = _build_deeplab
      model_args = (FLAGS, inputs_queue.dequeue(), outputs_to_num_classes, outputs_to_indices, bin_vals, bin_range, dataset, codes, True)
      clones = model_deploy.create_clones(config, model_fn, args=model_args)

      # Gather update_ops from the first clone. These contain, for example,
      # the updates for the batch_norm variables created by model_fn.
      first_clone_scope = config.clone_scope(0) # clone_0
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    with tf.device('/device:GPU:%d'%(FLAGS.num_clones+1)):
        if FLAGS.if_val:
          ## Construct the validation graph; takes one GPU.
          image_names, z_logits, outputs_to_weights, seg_one_hots_list, weights_normalized, car_nums, car_nums_list, idx_xys, reg_logits_pose_in_metric, pose_dict_N, prob_logits_pose, rotuvd_dict_N, masks_float, label_uv_flow_map, logits_uv_flow_map = _build_deeplab(FLAGS, inputs_queue_val.dequeue(), outputs_to_num_classes, outputs_to_indices, bin_vals, bin_range, dataset_val, codes, is_training=False)
          # pose_dict_N, xyz = _build_deeplab(FLAGS, inputs_queue_val.dequeue(), outputs_to_num_classes, outputs_to_indices, bin_vals, bin_range, dataset_val, codes, is_training=False)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for images, labels, semantic predictions
    summary_loss_dict = {}
    if FLAGS.save_summaries_images:
      if FLAGS.num_clones > 1:
          pattern_train = first_clone_scope + '/%s:0'
      else:
          pattern_train = '%s:0'
      pattern_val = 'val-%s:0'
      pattern = pattern_val if FLAGS.if_val else pattern_train
      # pattern = pattern_train

      gather_list = range(min(3, int(FLAGS.train_batch_size/FLAGS.num_clones)))
      print gather_list

      def scale_to_255(tensor, pixel_scaling=None):
          tensor = tf.to_float(tensor)
          if pixel_scaling == None:
              offset_to_zero = tf.reduce_min(tf.reduce_min(tf.reduce_min(tensor, axis=3, keepdims=True), axis=2, keepdims=True), axis=1, keepdims=True)
              scale_to_255 = tf.div(255., tf.reduce_max(tf.reduce_max(tf.reduce_max(
                  tensor - offset_to_zero, axis=3, keepdims=True), axis=2, keepdims=True), axis=1, keepdims=True))
              # offset_to_zero = tf.reduce_min(tensor)
              # scale_to_255 = tf.div(255., tf.reduce_max(tensor - offset_to_zero))
          else:
              offset_to_zero, scale_to_255 = pixel_scaling
          summary_tensor_float = tensor - offset_to_zero
          summary_tensor_float = summary_tensor_float * scale_to_255
          summary_tensor_float = tf.clip_by_value(summary_tensor_float, 0., 255.)
          summary_tensor_uint8 = tf.cast(summary_tensor_float, tf.uint8)
          return summary_tensor_uint8, (offset_to_zero, scale_to_255)
      x_coords = tf.range(68)
      y_coords = tf.range(170)
      Xs, Ys = tf.meshgrid(x_coords, y_coords)
      features_Ys = tf.tile(tf.expand_dims(tf.expand_dims(tf.transpose(Xs), -1), 0), [1, 1, 1, 1])
      features_Xs = tf.tile(tf.expand_dims(tf.expand_dims(tf.transpose(Ys), -1), 0), [1, 1, 1, 1])

      features_Xs_summary, _ = scale_to_255(features_Xs*4)
      features_Ys_summary, _ = scale_to_255(features_Ys*4)
      summaries.add(tf.summary.image('test/features_Xs', features_Xs_summary))
      summaries.add(tf.summary.image('test/features_Ys', features_Ys_summary))

      for pattern in [pattern_train, pattern_val] if FLAGS.if_val else [pattern_train]:
          if pattern == pattern_train:
              label_postfix = ''
          else:
              label_postfix = '_val'

          summary_mask = graph.get_tensor_by_name(pattern%'not_ignore_mask_in_loss')
          summary_mask = tf.reshape(summary_mask, [-1, dataset.height, dataset.width, 1])
          summary_mask_float = tf.to_float(summary_mask)
          summaries.add(tf.summary.image('gt'+label_postfix+'/%s' % 'not_ignore_mask', tf.gather(tf.cast(summary_mask_float*255., tf.uint8), gather_list)))

          mask_rescaled_float = graph.get_tensor_by_name(pattern%'mask_rescaled_float')

          seg_outputs = graph.get_tensor_by_name(pattern%'seg')
          summary_seg_output = tf.where(summary_mask, seg_outputs, tf.zeros_like(seg_outputs))
          summary_seg_output_uint8, _ = scale_to_255(summary_seg_output)
          summaries.add(tf.summary.image(
              'gt'+label_postfix+'/seg', tf.gather(summary_seg_output_uint8, gather_list)))

          summary_image = graph.get_tensor_by_name(pattern%common.IMAGE)
          summaries.add(tf.summary.image('gt'+label_postfix+'/%s' % common.IMAGE, tf.gather(summary_image, gather_list)))

          summary_image_name = graph.get_tensor_by_name(pattern%common.IMAGE_NAME)
          summaries.add(tf.summary.text('gt'+label_postfix+'/%s' % common.IMAGE_NAME, tf.gather(summary_image_name, gather_list)))

          summary_vis = graph.get_tensor_by_name(pattern%'vis')
          summaries.add(tf.summary.image('gt'+label_postfix+'/%s' % 'vis', tf.gather(summary_vis, gather_list)))

          # summary_rot_diffs = graph.get_tensor_by_name(pattern%'rot_error_map')
          # summary_rot_diffs = tf.where(summary_mask, summary_rot_diffs, tf.zeros_like(summary_rot_diffs))
          # summary_rot_diffs_uint8, _ = scale_to_255(summary_rot_diffs)
          # summaries.add(tf.summary.image('metrics_map'+label_postfix+'/%s' % 'rot_diffs', tf.gather(summary_rot_diffs_uint8, gather_list)))

          # summary_trans_diffs = graph.get_tensor_by_name(pattern%'trans_error_map')
          # summary_trans_diffs = tf.where(summary_mask, summary_trans_diffs, tf.zeros_like(summary_trans_diffs))
          # summary_trans_diffs_uint8, _ = scale_to_255(summary_trans_diffs)
          # summaries.add(tf.summary.image('metrics_map'+label_postfix+'/%s' % 'trans_diffs', tf.gather(summary_trans_diffs, gather_list)))

          if FLAGS.if_summary_shape_metrics:
              shape_id_sim_map_train = graph.get_tensor_by_name(pattern_train%'shape_id_sim_map')
              shape_id_sim_map_uint8_train, _ = scale_to_255(shape_id_sim_map_train, pixel_scaling=(0., 255.))
              summaries.add(tf.summary.image('metrics_map/shape_id_sim_map-trainInv', tf.gather(shape_id_sim_map_uint8_train, gather_list)))

              shape_id_sim_map = graph.get_tensor_by_name(pattern%'shape_id_sim_map')
              shape_id_sim_map_uint8, _ = scale_to_255(shape_id_sim_map, pixel_scaling=(0., 255.))
              summaries.add(tf.summary.image('metrics_map/shape_id_sim_map-valInv', tf.gather(shape_id_sim_map_uint8, gather_list)))


          label_uv_flow_map = graph.get_tensor_by_name(pattern%'label_uv_flow_map')
          logits_uv_flow_map = graph.get_tensor_by_name(pattern%'logits_uv_flow_map')
          for output_idx, output in enumerate(['u', 'v']):
              summary_label_output = tf.gather(label_uv_flow_map, [output_idx], axis=3)
              summary_label_output= tf.where(summary_mask, summary_label_output, tf.zeros_like(summary_label_output))
              summary_label_output_uint8, pixel_scaling = scale_to_255(summary_label_output)
              summaries.add(tf.summary.image('test'+label_postfix+'/%s_flow_label' % output, tf.gather(summary_label_output_uint8, gather_list)))

              summary_logits_output = tf.gather(logits_uv_flow_map, [output_idx], axis=3)
              summary_logits_output = mask_rescaled_float * summary_logits_output
              summary_logits_output_uint8, _ = scale_to_255(summary_logits_output, pixel_scaling)
              summaries.add(tf.summary.image('test'+label_postfix+'/%s_flow_logits' % output, tf.gather(summary_logits_output_uint8, gather_list)))

          for trans_metrics in ['trans_l2', 'depth_diff_abs', 'depth_relative', 'x_l1', 'y_l1']:
              if pattern == pattern_val:
                summary_trans = graph.get_tensor_by_name(pattern%trans_metrics)
              else:
                summary_trans = train_utils.get_avg_tensor_from_scopes(FLAGS.num_clones, '%s:0', graph, config, trans_metrics, return_concat=True)
              summaries.add(tf.summary.histogram('metrics_map'+label_postfix+'/%s' % trans_metrics, summary_trans))

          label_outputs = graph.get_tensor_by_name(pattern%'label_pose_shape_map')
          # label_id_outputs = graph.get_tensor_by_name(pattern%'pose_shape_label_id_map')
          logit_outputs = graph.get_tensor_by_name(pattern%'prob_logits_pose_shape_map')
          # seg_one_hots_outputs = graph.get_tensor_by_name(pattern%'seg_one_hots')
          for output_idx, output in enumerate(dataset.output_names):
              # # Scale up summary image pixel values for better visualization.
              summary_label_output = tf.gather(label_outputs, [output_idx], axis=3)
              summary_label_output= tf.where(summary_mask, summary_label_output, tf.zeros_like(summary_label_output))
              summary_label_output_uint8, pixel_scaling = scale_to_255(summary_label_output)
              summaries.add(tf.summary.image('output'+label_postfix+'/%s_label' % output, tf.gather(summary_label_output_uint8, gather_list)))

              summary_logits_output = tf.gather(logit_outputs, [output_idx], axis=3)
              summary_logits_output = tf.where(summary_mask, summary_logits_output, tf.zeros_like(summary_logits_output))
              summary_logits_output_uint8, _ = scale_to_255(summary_logits_output, pixel_scaling)
              summaries.add(tf.summary.image(
                  'output'+label_postfix+'/%s_logit' % output, tf.gather(summary_logits_output_uint8, gather_list)))

              summary_weights_output = graph.get_tensor_by_name(pattern%('%s_weights_map'%output))
              summary_weights_output = mask_rescaled_float * summary_weights_output
              summary_weights_output_uint8, _ = scale_to_255(summary_weights_output)
              summaries.add(tf.summary.image(
                  'output'+label_postfix+'/%s_weights' % output, tf.gather(summary_weights_output_uint8, gather_list)))

              # summary_seg_one_hots_output = tf.gather(seg_one_hots_outputs, [output_idx], axis=3)
              # summary_seg_one_hots_output_uint8, _ = scale_to_255(summary_seg_one_hots_output, pixel_scaling=(0., 255.))
              # summaries.add(tf.summary.image('test/%s_one_hot' % output, tf.gather(summary_seg_one_hots_output_uint8, gather_list)))

              # summary_label_id_output = tf.to_float(tf.gather(label_id_outputs, [output_idx], axis=3))
              # summary_label_id_output = tf.where(summary_mask, summary_label_id_output+1, tf.zeros_like(summary_label_id_output))
              # summary_label_id_output_uint8, _ = scale_to_255(summary_label_id_output)
              # summary_label_id_output_uint8 = tf.identity(summary_label_id_output_uint8, 'tttt'+output)
              # summaries.add(tf.summary.image(
              #     'test/%s_label_id' % output, tf.gather(summary_label_id_output_uint8, gather_list)))

              summary_diff = tf.abs(tf.to_float(summary_label_output_uint8) - tf.to_float(summary_logits_output_uint8))
              summary_diff = tf.where(summary_mask, summary_diff, tf.zeros_like(summary_diff))
              summaries.add(tf.summary.image('diff_map'+label_postfix+'/%s_ldiff' % output, tf.gather(tf.cast(summary_diff, tf.uint8), gather_list)))

              if output_idx in [0, 1, 2, 3, 4,5,6]:
                  summary_loss = graph.get_tensor_by_name((pattern%'loss_slice_reg_').replace(':0', '')+output+':0')
                  summaries.add(tf.summary.scalar('slice_loss'+label_postfix+'/'+(pattern%'reg_').replace(':0', '')+output, summary_loss))

                  if output_idx in [0, 1, 2, 3, 6]:
                      summary_loss = graph.get_tensor_by_name((pattern%'loss_slice_cls_').replace(':0', '')+output+':0')
                      summaries.add(tf.summary.scalar('slice_loss'+label_postfix+'/'+(pattern%'cls_').replace(':0', '')+output, summary_loss))

          add_metrics = ['loss_all_shape_id_cls_metric', 'loss_reg_shape'] if FLAGS.if_summary_shape_metrics else []
          for loss_name in ['loss_reg_rot_quat_metric', 'loss_reg_rot_quat', 'loss_reg_trans_metric', 'loss_reg_Zdepth_metric', 'loss_reg_Zdepth_relative_metric', 'loss_reg_x_metric', 'loss_reg_y_metric',
                  'loss_reg_trans', 'loss_cls_ALL'] + add_metrics:
              if pattern == pattern_val:
                summary_loss_avg = graph.get_tensor_by_name(pattern%loss_name)
              else:
                summary_loss_avg = train_utils.get_avg_tensor_from_scopes(FLAGS.num_clones, '%s:0', graph, config, loss_name)
              summaries.add(tf.summary.scalar(('total_loss%s/'%label_postfix+pattern%loss_name).replace(':0', ''), summary_loss_avg))


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
      last_layers = last_layers + ['decoder', 'decoder_weights', ]
      print '////last layers', last_layers

      # Keep trainable variables for last layers ONLY.
      # weight_scopes = [output_name+'_weights' for output_name in dataset.output_names] + ['decoder_weights']
      # grads_and_vars = train_utils.filter_gradients(weight_scopes, grads_and_vars)
      # print '==== variables_to_train: ', [grad_and_var[1].op.name for grad_and_var in grads_and_vars]

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
        loss= 0

        # calc training losses
        loss, should_stop = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)
        print loss
        # # print 'loss: ', loss
        # first_clone_test = graph.get_tensor_by_name(
                # ('%s/%s:0' % (first_clone_scope, 'z')).strip('/'))
        # test = sess.run(outputs_to_weights['z'])
        # # print test
        # print 'test: ', test.shape, np.max(test), np.min(test), np.mean(test), test.dtype

        # mask_rescaled_float = graph.get_tensor_by_name('%s:0'%'mask_rescaled_float')
        # test_out, test_out2 = sess.run([pose_dict_N, xyz])
        # print test_out
        # print test_out2
        # test_out3 = test_out3[test_out4!=0.]
        # print test_out3
        # print 'outputs_to_weights[z] masked: ', test_out3.shape, np.max(test_out3), np.min(test_out3), np.mean(test_out3), test_out3.dtype
        # print 'mask_rescaled_float: ', test_out4.shape, np.max(test_out4), np.min(test_out4), np.mean(test_out4), test_out4.dtype

        # test_1 = graph.get_tensor_by_name(
        #         ('%s/%s:0' % (first_clone_scope, 'prob_logits_pose')).strip('/'))
        # test_2 = graph.get_tensor_by_name(
        #         ('%s/%s:0' % (first_clone_scope, 'pose_dict_N')).strip('/'))
        # test_out, test_out2 = sess.run([test_1, test_2])
        # print '-- prob_logits_pose: ', test_out.shape, np.max(test_out), np.min(test_out), np.mean(test_out), test_out.dtype
        # print test_out, test_out.shape
        # print '-- pose_dict_N: ', test_out2.shape, np.max(test_out2), np.min(test_out2), np.mean(test_out2), test_out2.dtype
        # print test_out2, test_out2.shape

        should_stop = 0

        if FLAGS.if_val and train_step_fn.step % FLAGS.val_interval_steps == 0:
            # first_clone_test = graph.get_tensor_by_name('val-loss_all:0')
            # test = sess.run(first_clone_test)
            print '-- Validating...'
            # first_clone_test = graph.get_tensor_by_name(
            #         ('%s/%s:0' % (first_clone_scope, 'z')).strip('/'))
            # # first_clone_test2 = graph.get_tensor_by_name(
            # #         ('%s/%s:0' % (first_clone_scope, 'shape_id_sim_map')).strip('/'))
            # # first_clone_test3 = graph.get_tensor_by_name(
            # #         ('%s/%s:0' % (first_clone_scope, 'not_ignore_mask_in_loss')).strip('/'))

            mask_rescaled_float = graph.get_tensor_by_name('val-%s:0'%'mask_rescaled_float')
            _, test_out, test_out2, test_out3, test_out4, test_out5, test_out6, test_out7, test_out8, test_out9, test_out10, test_out11, test_out12, test_out13, test_out14, test_out15 = sess.run([summary_op, image_names, z_logits, outputs_to_weights['z'], mask_rescaled_float, weights_normalized, prob_logits_pose, pose_dict_N, car_nums, car_nums_list, idx_xys, rotuvd_dict_N, masks_float, reg_logits_pose_in_metric, label_uv_flow_map, logits_uv_flow_map])
            print test_out
            print test_out2.shape
            test_out3 = test_out3[test_out4!=0.]
            print 'outputs_to_weights[z] masked: ', test_out3.shape, np.max(test_out3), np.min(test_out3), np.mean(test_out3), test_out3.dtype
            print 'areas: ', test_out5.T, test_out5.shape, np.sum(test_out5)
            print 'masks: ', test_out12.T

            print '-- reg_logits_pose_in_metric: ', test_out13.shape, np.max(test_out13), np.min(test_out13), np.mean(test_out13), test_out13.dtype
            print test_out13, test_out13.shape
            print '-- pose_dict_N: ', test_out7.shape, np.max(test_out7), np.min(test_out7), np.mean(test_out7), test_out7.dtype
            print test_out7, test_out7.shape
            print '-- prob_logits_pose: ', test_out6.shape, np.max(test_out6), np.min(test_out6), np.mean(test_out6), test_out6.dtype
            print test_out6, test_out6.shape
            print '-- rotuvd_dict_N: ', test_out11.shape, np.max(test_out11), np.min(test_out11), np.mean(test_out11), test_out11.dtype
            print test_out11, test_out11.shape
            print '-- label_uv_flow_map: ', test_out14.shape, np.max(test_out14[:, :, :, 0]), np.min(test_out14[:, :, :, 0]), np.max(test_out14[:, :, :, 1]), np.min(test_out14[:, :, :, 1])
            print '-- logits_uv_flow_map: ', test_out15.shape, np.max(test_out15[:, :, :, 0]), np.min(test_out14[:, :, :, 0]), np.max(test_out15[:, :, :, 0]), np.min(test_out15[:, :, :, 0])
            print '-- car_nums: ', test_out8, test_out9, test_out10.T

            # # Vlen(test_out), test_out[0].shape
            # # print test_out2.shape, test_out2
            # # print test_out3
            # # print test_out, test_out.shape
            # # # test_out = test[:, :, :, 3]
            # # test_out = test_out[test_out3]
            # # # test_out2 = test2[:, :, :, 3]
            # # test_out2 = test_out2[test_out3]
            # # # print test_out
            # # print 'shape_id_map: ', test_out.shape, np.max(test_out), np.min(test_out), np.mean(test_out), np.median(test_out), test_out.dtype
            # # print 'shape_id_sim_map: ', test_out2.shape, np.max(test_out2), np.min(test_out2), np.mean(test_out2), np.median(test_out2), test_out2.dtype
            # # print 'masks sum: ', test_out3.dtype, np.sum(test_out3.astype(float))
            # # assert np.max(test_out) == np.max(test_out2), 'MAtch1!!!'
            # # assert np.min(test_out) == np.min(test_out2), 'MAtch2!!!'

        return [loss, should_stop]
    train_step_fn.step = 0


    # trainables = [v.name for v in tf.trainable_variables()]
    # alls =[v.name for v in tf.all_variables()]
    # print '----- Trainables %d: '%len(trainables), trainables
    # print '----- All %d: '%len(alls), alls
    # print '===== ', len(list(set(trainables) - set(alls)))
    # print '===== ', len(list(set(alls) - set(trainables))), list(set(alls) - set(trainables))
    # print summaries

    if FLAGS.if_print_tensors:
        for op in tf.get_default_graph().get_operations():
            print str(op.name)

    init_assign_op, init_feed_dict = train_utils.model_init(
        FLAGS.restore_logdir,
        FLAGS.tf_initial_checkpoint,
        FLAGS.if_restore,
        FLAGS.initialize_last_layer,
        last_layers,
        # ignore_including=['_weights/BatchNorm', 'decoder_weights'],
        ignore_including=None,
        ignore_missing_vars=True)

    def InitAssignFn(sess):
        sess.run(init_assign_op, init_feed_dict)

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
        init_fn=InitAssignFn,
        # init_fn=train_utils.get_model_init_fn(
        #     FLAGS.restore_logdir,
        #     FLAGS.tf_initial_checkpoint,
        #     FLAGS.if_restore,
        #     FLAGS.initialize_last_layer,
        #     last_layers,
        #     ignore_missing_vars=True),
        summary_op=summary_op,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
  flags.mark_flag_as_required('base_logdir')
  flags.mark_flag_as_required('tf_initial_checkpoint')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
