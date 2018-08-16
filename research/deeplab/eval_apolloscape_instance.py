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
"""evaluation script for the DeepLab model.

See model.py for more details and usage.
"""
import warnings
warnings.filterwarnings("ignore")
import math
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

# Settings for logging.

flags.DEFINE_string('task_name', 'tmp',
                    'Task name; will be appended to FLAGS.eval_logdir to log files.')

flags.DEFINE_string('restore_name', None,
                    'Task name to restore; will be appended to FLAGS.eval_logdir to log files.')

flags.DEFINE_string('base_logdir', None,
                    'Where the checkpoint and logs are stored (base dir).')

flags.DEFINE_string('eval_logdir', None,
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_string('restore_logdir', None,
                    'Where the checkpoint and logs are REstored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('max_number_of_evaluations', 1,
        'Maximum number of eval iterations. Will loop '
        'indefinitely upon nonpositive values.')

flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as images to summary.')

# Settings for evaluation strategy.
flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')

# For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
flags.DEFINE_boolean('if_discrete_loss', True,
                     'Use discrete regression + classification loss.')

flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during evaluation.')

# Settings for fine-tuning the network.

flags.DEFINE_boolean('if_restore', True,
                    'Whether to restore the logged checkpoint.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_feature_extractor', True,
                     'Fine tune the feature extractors or not.')

flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during evaluation/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.
flags.DEFINE_string('dataset', 'apolloscape',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset to be used for evaluation.')

flags.DEFINE_string('dataset_dir', 'deeplab/datasets/apolloscape', 'Where the dataset reside.')

from build_deeplab import _build_deeplab

def main(unused_argv):
  FLAGS.restore_logdir = FLAGS.base_logdir + '/' + FLAGS.restore_name
  FLAGS.eval_logdir = FLAGS.base_logdir + '/' + FLAGS.task_name + '/' + 'eval'
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('==== Logging in dir:%s; Evaluating on %s set', FLAGS.eval_logdir, FLAGS.eval_split)

  # Get dataset-dependent information.
  dataset = regression_dataset.get_dataset(
      FLAGS.dataset, FLAGS.eval_split, dataset_dir=FLAGS.dataset_dir)
  print '#### The data has size:', dataset.num_samples

  with tf.Graph().as_default() as graph:
    bin_range = [np.linspace(r[0], r[1], num=b).tolist() for r, b in zip(dataset.pose_range, dataset.bin_nums[:-1])]
    outputs_to_num_classes = {}
    outputs_to_indices = {}
    for output, bin_num, idx in zip(dataset.output_names, dataset.bin_nums,range(len(dataset.output_names))):
        if FLAGS.if_discrete_loss:
          outputs_to_num_classes[output] = bin_num
        else:
         outputs_to_num_classes[output] = 1
        outputs_to_indices[output] = idx
    bin_vals = [tf.constant(value=[bin_range[i]], dtype=tf.float32, shape=[1, dataset.bin_nums[i]], name=name) \
            for i, name in enumerate(dataset.output_names[:-1])]

    samples = input_generator.get(
        dataset,
        FLAGS.eval_batch_size,
        dataset_split=FLAGS.eval_split,
        is_training=False,
        model_variant=FLAGS.model_variant,
        num_epochs=None if FLAGS.max_number_of_evaluations>=1 else 1)

    # Create the global step on the device storing the variables.
      ## Construct the validation graph; takes one GPU.
    _build_deeplab(FLAGS, samples, outputs_to_num_classes, outputs_to_indices, bin_vals, dataset, is_training=False, reuse=False)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for images, labels, semantic predictions
    pattern = 'val-%s:0'
    if FLAGS.save_summaries_images:

      summary_mask = graph.get_tensor_by_name(pattern%'not_ignore_mask_in_loss')
      summary_mask = tf.reshape(summary_mask, [-1, dataset.height, dataset.width, 1])
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

      summary_rot_diffs = graph.get_tensor_by_name(pattern%'rot_diffs')
      summaries.add(tf.summary.image('diff_map/%s' % 'rot_diffs', tf.gather(summary_rot_diffs, [0, 1, 2])))

      summary_trans_diffs = graph.get_tensor_by_name(pattern%'trans_diffs')
      summaries.add(tf.summary.image('diff_map/%s' % 'trans_diffs', tf.gather(summary_trans_diffs, [0, 1, 2])))

      for output_idx, output in enumerate(dataset.output_names[:-1]):
          # # Scale up summary image pixel values for better visualization.
          summary_label_output = tf.gather(label_outputs, [output_idx], axis=3)
          summary_label_output= tf.where(summary_mask, summary_label_output, tf.zeros_like(summary_label_output))
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

      summary_loss = graph.get_tensor_by_name(pattern%'loss_all')
      summaries.add(tf.summary.scalar(('total_loss/'+pattern%'loss_all').replace(':0', ''), summary_loss))

      summary_loss_rot = graph.get_tensor_by_name(pattern%'loss_all_rot_quat_metric')
      summaries.add(tf.summary.scalar(('total_loss/'+pattern%'loss_all_rot_quat_metric').replace(':0', ''), summary_loss_rot))

      summary_loss_rot = graph.get_tensor_by_name(pattern%'loss_all_rot_quat')
      summaries.add(tf.summary.scalar(('total_loss/'+pattern%'loss_all_rot_quat').replace(':0', ''), summary_loss_rot))

      summary_loss_rot = graph.get_tensor_by_name(pattern%'loss_all_trans_metric')
      summaries.add(tf.summary.scalar(('total_loss/'+pattern%'loss_all_trans_metric').replace(':0', ''), summary_loss_rot))

      summary_loss_trans = graph.get_tensor_by_name(pattern%'loss_all_trans')
      summaries.add(tf.summary.scalar(('total_loss/'+pattern%'loss_all_trans').replace(':0', ''), summary_loss_trans))

      summary_loss_cls_all = graph.get_tensor_by_name(pattern%'loss_cls_ALL')
      summaries.add(tf.summary.scalar(('total_loss/'+pattern%'loss_cls_ALL').replace(':0', ''), summary_loss_cls_all))

      summary_loss_shape = graph.get_tensor_by_name(pattern%'loss_all_shape')
      summaries.add(tf.summary.scalar(('total_loss/'+pattern%'loss_all_shape').replace(':0', ''), summary_loss_shape))


    # Merge all summaries together.
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summary_op = tf.summary.merge(list(summaries))

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    def train_step_fn(sess, train_op, global_step, train_step_kwargs):
        train_step_fn.step += 1  # or use global_step.eval(session=sess)

        # calc evaluation losses
        loss, should_stop = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)
        print loss
        # print 'loss: ', loss
        # first_clone_test = graph.get_tensor_by_name(
        #         ('%s/%s:0' % (first_clone_scope, 'shape_map')).strip('/'))
        # test = sess.run(first_clone_test)
        # # print test
        # print 'test: ', test.shape, np.max(test), np.min(test), np.mean(test), test.dtype
        should_stop = 0

        if FLAGS.if_val and train_step_fn.step % FLAGS.val_interval_steps == 0:
            first_clone_test = graph.get_tensor_by_name('val-loss_all:0')
            test = sess.run(first_clone_test)
            print '-- Validating... Loss: %.4f'%test
            first_clone_test = graph.get_tensor_by_name(
                    ('%s/%s:0' % (first_clone_scope, 'scaled_shape_logits')).strip('/'))
            first_clone_test2 = graph.get_tensor_by_name(
                    ('%s/%s:0' % (first_clone_scope, 'shape_map')).strip('/'))
                    # 'ttttrow:0')
            test, test2 = sess.run([first_clone_test, first_clone_test2])
            test_out = test[:, :, :, 3]
            test_out2 = test2[:, :, :, 3]
            # print test_out
            print 'output: ', test_out.shape, np.max(test_out), np.min(test_out), np.mean(test_out), np.median(test_out), test_out.dtype
            print 'label: ', test_out2.shape, np.max(test_out2), np.min(test_out2), np.mean(test_out2), np.median(test_out2), test_out2.dtype

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
        # # print 'evaluation....... logits stats: ', np.max(logits), np.min(logits), np.mean(logits)
        # # label_one_piece = label[0, :, :, 0]
        # # print 'evaluation....... label stats', np.max(label_one_piece), np.min(label_one_piece), np.sum(label_one_piece[label_one_piece!=255.])
        return [loss, should_stop]

    # # Define the evaluation metric.
    # metric_map = {}
    # predictions_tag = 'loss_all_trans_metric'
    # metric_map[predictions_tag] = [graph.get_tensor_by_name(pattern%'loss_all_trans_metric')]
    # metrics_to_values, metrics_to_updates = (
    #         tf.contrib.metrics.aggregate_metric_map(metric_map)
    #         )
    num_batches = int(math.ceil(dataset.num_samples / float(FLAGS.eval_batch_size)))
    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:
        num_eval_iters = FLAGS.max_number_of_evaluations
    metric = tf.metrics.mean_squared_error(
            graph.get_tensor_by_name(pattern%common.LABEL),
            graph.get_tensor_by_name(pattern%'scaled_logits'))
    print metric
    names_to_values, names_to_updates = tf.contrib.metrics.aggregate_metric_map({'mse': metric})

    # Start the evaluation.
    metric_values = slim.evaluation.evaluate_once(
            master='',
            checkpoint_path=tf.train.latest_checkpoint(FLAGS.restore_logdir),
            logdir=FLAGS.eval_logdir,
            num_evals=num_batches,
            eval_op=names_to_updates.values(),
            final_op=names_to_updates.values(),
            )
    # metric_values = slim.evaluation.evaluation_loop(
    #         master='',
    #         checkpoint_dir=FLAGS.restore_logdir,
    #         logdir=FLAGS.eval_logdir,
    #         num_evals=num_batches,
    #         eval_op=names_to_updates.values(),
    #         final_op=names_to_updates.values(),
    #         max_number_of_evaluations=num_eval_iters,
    #         eval_interval_secs=100,
    #         )
    for metric, value in zip(names_to_values.keys(), metric_values):
        print 'Metric %s has value: %f', metric, value
    # slim.evaluation.evaluation_loop(
    #         # checkpoint_dir=FLAGS.checkpoint_dir,
    #         logdir=FLAGS.eval_logdir,
    #         num_evals=num_batches,
    #         eval_op=list(metrics_to_updates.values()),
    #         max_number_of_evaluations=num_eval_iters,
    #         # eval_interval_secs=FLAGS.eval_interval_secs,
    #         init_fn=train_utils.get_model_init_fn(
    #             FLAGS.restore_logdir,
    #             tf_initial_checkpoint=None,
    #             restore_logged=FLAGS.if_restore,
    #             initialize_last_layer=False,
    #             last_layers=None,
    #             ignore_missing_vars=False),
    #         )

if __name__ == '__main__':
  flags.mark_flag_as_required('base_logdir')
  flags.mark_flag_as_required('restore_name')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
