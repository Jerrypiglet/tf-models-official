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
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import regression_dataset
from deeplab.utils import input_generator
from deeplab.utils import train_utils
from deployment import model_deploy
import numpy as np
np.set_printoptions(threshold=np.nan)

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

flags.DEFINE_string('train_logdir', None,
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_interval_secs', 300,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 30,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')

# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 30000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('train_batch_size', 8,
                     'The number of images in each batch during training.')

# For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
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

flags.DEFINE_string('dataset_dir', 'deeplab/datasets/apolloscape', 'Where the dataset reside.')


def _build_deeplab(inputs_queue, outputs_to_num_classes, ignore_label):
  """Builds a clone of DeepLab.

  Args:
    inputs_queue: A prefetch queue for images and labels.
    # outputs_to_num_classes: A map from output type to the number of classes.
    #   For example, for the task of semantic segmentation with 21 semantic
    #   classes, we would have outputs_to_num_classes['semantic'] = 21.
    ignore_label: Ignore label.

  Returns:
    A map of maps from output_type (e.g., semantic prediction) to a
      dictionary of multi-scale logits names to logits. For each output_type,
      the dictionary has keys which correspond to the scales and values which
      correspond to the logits. For example, if `scales` equals [1.0, 1.5],
      then the keys would include 'merged_logits', 'logits_1.00' and
      'logits_1.50'.
  """
  samples = inputs_queue.dequeue()

  # Add name to input and label nodes so we can add to summary.
  samples[common.IMAGE] = tf.identity(
      samples[common.IMAGE], name=common.IMAGE)
  samples[common.LABEL] = tf.identity(
      samples[common.LABEL], name=common.LABEL)
  samples['original_label'] = tf.identity(samples['original_label'], name='original_label')

  model_options = common.ModelOptions(
      outputs_to_num_classes=outputs_to_num_classes,
      crop_size=FLAGS.train_crop_size,
      atrous_rates=FLAGS.atrous_rates,
      output_stride=FLAGS.output_stride)

  outputs_to_scales_to_logits = model.multi_scale_logits(
      samples[common.IMAGE],
      model_options=model_options,
      image_pyramid=FLAGS.image_pyramid,
      weight_decay=FLAGS.weight_decay,
      is_training=True,
      fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
      fine_tune_feature_extractor=FLAGS.fine_tune_feature_extractor)

  # print outputs_to_scales_to_logits, 'outputs_to_scales_to_logits @_build_deeplab @train_apolloscape_instance.py' # {'regression': {'merged_logits': <tf.Tensor 'ResizeBilinear_2:0' shape=(4, 49, 49, 6) dtype=float32>}}

  # Add name to graph node so we can add to summary.
  output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
  output_type_dict[model.MERGED_LOGITS_SCOPE] = tf.identity(
      output_type_dict[model.MERGED_LOGITS_SCOPE],
      name=common.OUTPUT_TYPE)
  # print output_type_dict, 'output_type_dict @_build_deeplab @train_apolloscape_instance.py' # {'merged_logits': <tf.Tensor 'regression:0' shape=(4, 49, 49, 6) dtype=float32>}

  for output, num_classes in six.iteritems(outputs_to_num_classes):
      # print output, num_classes, samples[common.LABEL], samples[common.IMAGE], '_build_deeplab@train_apolloscape_instance.py' # regression 6 Tensor("label:0", shape=(4, 769, 769, 6), dtype=float32), Tensor("image:0", shape=(4, 769, 769, 3), dtype=float32)
      # print '--- outputs_to_scales_to_logits[output]', outputs_to_scales_to_logits[output]
      not_ignore_mask, scaled_logits = train_utils.add_regression_l2_loss_for_each_scale(
        outputs_to_scales_to_logits[output], # {'merged_logits': <tf.Tensor 'regression:0' shape=(4, 49, 49, 6) dtype=float32>}
        samples[common.LABEL],
        # num_classes,
        ignore_label,
        loss_weight=1.0,
        upsample_logits=FLAGS.upsample_logits,
        scope=output)
      scaled_logits = tf.identity(scaled_logits, name='scaled_logits')
      not_ignore_mask = tf.identity(not_ignore_mask, name='not_ignore_mask')

  return outputs_to_scales_to_logits


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
  config = model_deploy.DeploymentConfig(
      num_clones=FLAGS.num_clones,
      clone_on_cpu=FLAGS.clone_on_cpu,
      replica_id=FLAGS.task,
      num_replicas=FLAGS.num_replicas,
      num_ps_tasks=FLAGS.num_ps_tasks)

  # Split the batch across GPUs.
  assert FLAGS.train_batch_size % config.num_clones == 0, (
      'Training batch size not divisble by number of clones (GPUs).')

  clone_batch_size = FLAGS.train_batch_size // config.num_clones

  # Get dataset-dependent information.
  dataset = regression_dataset.get_dataset(
      FLAGS.dataset, FLAGS.train_split, dataset_dir=FLAGS.dataset_dir)

  tf.gfile.MakeDirs(FLAGS.train_logdir)
  tf.logging.info('Training on %s set', FLAGS.train_split)

  with tf.Graph().as_default() as graph:
    with tf.device(config.inputs_device()):
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

    # Create the global step on the device storing the variables.
    with tf.device(config.variables_device()):
      global_step = tf.train.get_or_create_global_step()

      # Define the model and create clones.
      model_fn = _build_deeplab
      model_args = (inputs_queue, {common.OUTPUT_TYPE: dataset.num_classes}, dataset.ignore_label)
      clones = model_deploy.create_clones(config, model_fn, args=model_args)

      # Gather update_ops from the first clone. These contain, for example,
      # the updates for the batch_norm variables created by model_fn.
      first_clone_scope = config.clone_scope(0)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for model variables.
    # for model_var in slim.get_model_variables():
    #   summaries.add(tf.summary.histogram(model_var.op.name, model_var))

    # Add summaries for images, labels, semantic predictions
    if FLAGS.save_summaries_images:
      summary_mask = graph.get_tensor_by_name(
          ('%s/%s:0' % (first_clone_scope, 'not_ignore_mask')).strip('/'))
      summary_mask = tf.reshape(summary_mask, [-1, FLAGS.train_crop_size[0], FLAGS.train_crop_size[1], dataset.num_classes])
      if dataset.num_classes > 1:
          summary_mask = tf.gather(summary_mask, [0, 1, 2], axis=3)
      # summaries.add(tf.summary.image('samples/%s' % 'not_ignore_mask', tf.cast(summary_mask*255., tf.uint8)))

      summary_image = graph.get_tensor_by_name(
          ('%s/%s:0' % (first_clone_scope, common.IMAGE)).strip('/'))
      summaries.add(tf.summary.image('samples/%s' % common.IMAGE, summary_image))

      summary_label = graph.get_tensor_by_name(
          ('%s/%s:0' % (first_clone_scope, common.LABEL)).strip('/'))
      # # Scale up summary image pixel values for better visualization.
      if dataset.num_classes > 1:
          summary_label = tf.gather(first_clone_label, [0, 1, 2], axis=3)
      pixel_scaling = tf.div(255., tf.reduce_max(tf.where(tf.not_equal(summary_label, 255.), summary_label, tf.zeros_like(summary_label))))
      summary_label = tf.where(tf.equal(summary_mask, 1.), summary_label, tf.zeros_like(summary_label))
      summary_label_uint8 = tf.cast(summary_label * pixel_scaling, tf.uint8)
      summaries.add(tf.summary.image('samples/%s' % common.LABEL, summary_label_uint8))

      def scale_to_255(tensor, pixel_scaling=None):
          if pixel_scaling == None:
              new_pixel_scaling = tf.div(255., tf.reduce_max(tensor))
          else:
              new_pixel_scaling = pixel_scaling
          summary_tensor_uint8 = tf.cast(tensor * new_pixel_scaling, tf.uint8)
          return summary_tensor_uint8

      summary_regression = graph.get_tensor_by_name(
          ('%s/scaled_logits:0' % first_clone_scope).strip('/'))
      # predictions = tf.expand_dims(tf.argmax(first_clone_output, 3), -1)
      # summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
      if dataset.num_classes > 1:
          summary_regression = tf.gather(summary_regression, [0, 1, 2], axis=3)
      summary_regression = tf.where(tf.equal(summary_mask, 1.), summary_regression, tf.zeros_like(summary_regression))
      summaries.add(tf.summary.image(
          'samples/%s' % 'scaled_logits', scale_to_255(summary_regression, pixel_scaling)))

      summary_diff = tf.abs(summary_label - summary_regression)
      summary_diff = tf.where(tf.equal(summary_mask, 1.), summary_diff, tf.zeros_like(summary_diff))
      summaries.add(tf.summary.image('samples/%s' % 'diff', scale_to_255(summary_diff, pixel_scaling)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

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
    # for variable in slim.get_model_variables():
    #   summaries.add(tf.summary.histogram(variable.op.name, variable))

    with tf.device(config.variables_device()):
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, optimizer)
      # total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
      summaries.add(tf.summary.scalar('total_loss', total_loss))

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
        # print '////grad_mult', grad_mult
        print '////grads_and_vars', len(grads_and_vars), grads_and_vars

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
        # loss, should_stop = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)
        # print 'loss: ', loss
        should_stop = 0

        first_clone_label = graph.get_tensor_by_name(
                ('%s/%s:0' % (first_clone_scope, 'original_label')).strip('/'))
        first_clone_logit = graph.get_tensor_by_name(
                ('%s/%s:0' % (first_clone_scope, 'regression')).strip('/'))
        not_ignore_mask = graph.get_tensor_by_name(
                ('%s/%s:0' % (first_clone_scope, 'not_ignore_mask')).strip('/'))
        label, logits, mask = sess.run([first_clone_label, first_clone_logit, not_ignore_mask])
        # mask = np.reshape(mask, (-1, FLAGS.train_crop_size[0], FLAGS.train_crop_size[1], dataset.num_classes))

        # print '... shapes, types, loss', label.shape, label.dtype, logits.shape, logits.dtype, loss
        # print 'mask', mask.shape, np.mean(mask)
        print 'label', label.shape, np.mean(label)
        # # print 'training....... logits stats: ', np.max(logits), np.min(logits), np.mean(logits)
        # # label_one_piece = label[0, :, :, 0]
        # # print 'training....... label stats', np.max(label_one_piece), np.min(label_one_piece), np.sum(label_one_piece[label_one_piece!=255.])
        return [loss, should_stop]
    train_step_fn.step = 0


    trainables = [v.name for v in tf.trainable_variables()]
    alls =[v.name for v in tf.all_variables()]
    print '----- Trainables %d: '%len(trainables), trainables[:10]
    print '----- All %d: '%len(alls), alls[:10]
    print '===== ', len(list(set(trainables) - set(alls)))
    print '===== ', len(list(set(alls) - set(trainables)))

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
            FLAGS.train_logdir,
            FLAGS.tf_initial_checkpoint,
            FLAGS.initialize_last_layer,
            last_layers,
            ignore_missing_vars=True),
        summary_op=summary_op,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
  flags.mark_flag_as_required('train_logdir')
  flags.mark_flag_as_required('tf_initial_checkpoint')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
