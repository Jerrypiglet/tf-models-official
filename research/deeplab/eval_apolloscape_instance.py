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
import time
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
from scipy.io import savemat, loadmat
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

flags.DEFINE_boolean('if_depth', False,
        'True: regression to depth; False: regression to invd.')

flags.DEFINE_boolean('if_log_depth', False,
        'True: log depth space.')

flags.DEFINE_boolean('if_shape', True,
        'True: adding shape loss. if FLAGS.if_uvflow else None')

flags.DEFINE_boolean('if_uvflow', False,
        'True: regression to uv flow; False: regression to xy.')

flags.DEFINE_boolean('if_depth_only', False,
        'True: regression to depth only.')

# Dataset settings.
flags.DEFINE_string('dataset', 'apolloscape',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('val_split', 'val',
                    'Which split of the dataset to be used for evaluation.')

flags.DEFINE_string('dataset_dir', 'deeplab/datasets/apolloscape', 'Where the dataset reside.')

from build_deeplab import _build_deeplab

def main(unused_argv):
  FLAGS.restore_logdir = FLAGS.base_logdir + '/' + FLAGS.restore_name
  # FLAGS.eval_logdir = FLAGS.base_logdir + '/' + FLAGS.task_name + '/' + 'eval'
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('==== Logging in dir:%s; Evaluating on %s set', FLAGS.eval_logdir, FLAGS.val_split)

  # Get dataset-dependent information.
  dataset = regression_dataset.get_dataset(
      FLAGS.dataset, FLAGS.val_split, dataset_dir=FLAGS.dataset_dir)
  print '#### The data has size:', dataset.num_samples

  with tf.Graph().as_default() as graph:
    codes = np.load('/ssd2/public/zhurui/Documents/mesh-voxelization/models/cars_64/codes.npy')
    codes_max = np.amax(codes, axis=1).reshape((-1, 1))
    codes_min = np.amin(codes, axis=1).reshape((-1, 1))
    shape_range = np.hstack((codes_max + (codes_max - codes_min)/(dataset.SHAPE_BINS-1.), codes_min - (codes_max - codes_min)/(dataset.SHAPE_BINS-1.)))
    bin_range = [np.linspace(r[0], r[1], num=b).tolist() for r, b in zip(np.vstack((dataset.pose_range, shape_range)), dataset.bin_nums)]

    outputs_to_num_classes = {}
    outputs_to_indices = {}
    for output, bin_num, idx in zip(dataset.output_names, dataset.bin_nums,range(len(dataset.output_names))):
        outputs_to_num_classes[output] = bin_num
        outputs_to_indices[output] = idx
    bin_vals = [tf.constant(value=[bin_range[i]], dtype=tf.float32, shape=[1, dataset.bin_nums[i]], name=name) \
            for i, name in enumerate(dataset.output_names)]

    samples = input_generator.get(
        dataset,
        codes,
        FLAGS.eval_batch_size,
        dataset_split=FLAGS.val_split,
        is_training=False,
        model_variant=FLAGS.model_variant,
        num_epochs=None if FLAGS.max_number_of_evaluations>=1 else 1)

    # Create the global step on the device storing the variables.
      ## Construct the validation graph; takes one GPU.
    _build_deeplab(FLAGS, samples, outputs_to_num_classes, outputs_to_indices, bin_vals, bin_range, dataset, codes, is_training=False, reuse=False)

    # Add summaries for images, labels, semantic predictions
    pattern = 'val-%s:0'

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    # Define the evaluation metric.
    num_batches = int(math.ceil(dataset.num_samples / float(FLAGS.eval_batch_size)))
    num_eval_iters = None
    # metric = tf.metrics.mean_squared_error(
    #         graph.get_tensor_by_name(pattern%'label_pose_shape_map'),
    #         graph.get_tensor_by_name(pattern%'scaled_prob_logits_pose_shape_map'))
    # print metric
    # names_to_values, names_to_updates = tf.contrib.metrics.aggregate_metric_map({'mse': metric})

    # For single evaluation across the dataset.
    # metric_values = slim.evaluation.evaluate_once(
    #         master='',
    #         checkpoint_path=tf.train.latest_checkpoint(FLAGS.restore_logdir),
    #         logdir=FLAGS.eval_logdir,
    #         num_evals=num_batches,
    #         eval_op=names_to_updates.values(),
    #         final_op=names_to_updates.values(),
    #         )

    # For continuous timed evalatiion on new checkpoint.
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
    # for metric, value in zip(names_to_values.keys(), metric_values):
    #     print 'Metric %s has value: %f', metric, value

    shape_sim_mat = np.loadtxt('./deeplab/dataset-api/car_instance/sim_mat.txt')
    def _process_batch(sess, batch):
        # Label and outputs for pose and shape
        image = graph.get_tensor_by_name(pattern%common.IMAGE)
        seg = graph.get_tensor_by_name(pattern%'seg')
        logits_pose_shape_map = graph.get_tensor_by_name(pattern%'scaled_prob_logits_pose_shape_map')
        # For logging
        image_name = graph.get_tensor_by_name(pattern%common.IMAGE_NAME)
        mask = graph.get_tensor_by_name(pattern%'not_ignore_mask_in_loss')
        if FLAGS.val_split != 'test':
            label_pose_shape_map = graph.get_tensor_by_name(pattern%'label_pose_shape_map')
            vis = graph.get_tensor_by_name(pattern%'vis')
            shape_id_map = graph.get_tensor_by_name(pattern%'shape_id_map')
            shape_id_map_predict = graph.get_tensor_by_name(pattern%'shape_id_map_predict')
            # The metrics map
            rot_error_map = graph.get_tensor_by_name(pattern%'rot_error_map')
            trans_error_map = graph.get_tensor_by_name(pattern%'trans_error_map')
            shape_id_sim_map = graph.get_tensor_by_name(pattern%'shape_id_sim_map')
            image_out, vis_out, seg_out, \
                    label_pose_shape_map_out, logits_pose_shape_map_out, shape_id_map_out, shape_id_map_predict_out, \
                    rot_error_map_out, trans_error_map_out, shape_id_sim_map_out, \
                    image_name_out, mask_out = sess.run([image, vis, seg, \
                    label_pose_shape_map, logits_pose_shape_map, shape_id_map, shape_id_map_predict,  \
                    rot_error_map, trans_error_map, shape_id_sim_map, \
                    image_name, mask])
            print image_name_out
            savemat(FLAGS.eval_logdir+'/%d-%s.mat'%(batch, image_name_out[0]), {'image': image_out, 'vis': vis_out, 'seg': seg_out, \
                    'label_pose_shape_map': label_pose_shape_map_out, 'logits_pose_shape_map': logits_pose_shape_map_out, 'shape_id_map': shape_id_map_out, 'shape_id_map_predict': shape_id_map_predict_out, \
                    'rot_error_map': rot_error_map_out, 'trans_error_map': trans_error_map_out, 'shape_id_sim_map': shape_id_sim_map_out, \
                    'image_name': image_name_out, 'mask': mask_out})
        else:
            image_out, seg_out, logits_pose_shape_map_out, image_name_out, mask_out = \
                    sess.run([image, seg, logits_pose_shape_map, image_name, mask])
            print image_name_out
            savemat(FLAGS.eval_logdir+'/%d-%s.mat'%(batch, image_name_out[0]), {'image': image_out, 'seg': seg_out, 'logits_pose_shape_map': logits_pose_shape_map_out, 'image_name': image_name_out, 'mask': mask_out})

    tf.train.get_or_create_global_step()
    saver = tf.train.Saver(slim.get_variables_to_restore())
    sv = tf.train.Supervisor(graph=graph,
            logdir=FLAGS.eval_logdir,
            init_op=tf.global_variables_initializer(),
            summary_op=None,
            summary_writer=None,
            global_step=None,
            saver=saver)
    last_checkpoint = None

    # Loop to visualize the results when new checkpoint is created.
    num_iters = 0
    while (FLAGS.max_number_of_evaluations <= 0 or
            num_iters < FLAGS.max_number_of_evaluations):
        num_iters += 1
        last_checkpoint = slim.evaluation.wait_for_new_checkpoint(
                FLAGS.restore_logdir, last_checkpoint)
        start = time.time()
        # tf.logging.info('Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',time.gmtime()))
        tf.logging.info('Visualizing with model %s', last_checkpoint)

        with sv.managed_session('',
                start_standard_services=False) as sess:
            sv.start_queue_runners(sess)
            sv.saver.restore(sess, last_checkpoint)

            image_id_offset = 0
            # for batch in range(2):
            for batch in range(num_batches):
                tf.logging.info('Visualizing batch %d / %d', batch + 1, num_batches)

                _process_batch(sess, batch)

                image_id_offset += FLAGS.eval_batch_size


if __name__ == '__main__':
  flags.mark_flag_as_required('base_logdir')
  flags.mark_flag_as_required('restore_name')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
