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
r"""Provides DeepLab model definition and helper functions.

DeepLab is a deep learning system for semantic image segmentation with
the following features:

(1) Atrous convolution to explicitly control the resolution at which
feature responses are computed within Deep Convolutional Neural Networks.

(2) Atrous spatial pyramid pooling (ASPP) to robustly segment objects at
multiple scales with filters at multiple sampling rates and effective
fields-of-views.

(3) ASPP module augmented with image-level feature and batch normalization.

(4) A simple yet effective decoder module to recover the object boundaries.

See the following papers for more details:

"Encoder-Decoder with Atrous Separable Convolution for Semantic Image
Segmentation"
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.
(https://arxiv.org/abs/1802.02611)

"Rethinking Atrous Convolution for Semantic Image Segmentation,"
Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
(https://arxiv.org/abs/1706.05587)

"DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
Atrous Convolution, and Fully Connected CRFs",
Liang-Chieh Chen*, George Papandreou*, Iasonas Kokkinos, Kevin Murphy,
Alan L Yuille (* equal contribution)
(https://arxiv.org/abs/1606.00915)

"Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected
CRFs"
Liang-Chieh Chen*, George Papandreou*, Iasonas Kokkinos, Kevin Murphy,
Alan L. Yuille (* equal contribution)
(https://arxiv.org/abs/1412.7062)
"""
import tensorflow as tf
from deeplab.core import feature_extractor
from deeplab.utils import train_utils_mP as train_utils
from deeplab import common

slim = tf.contrib.slim

LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'
WEIGHTS_DECODER_SCOPE = 'decoder_weights'


def get_extra_layer_scopes(last_layers_contain_logits_only=False):
  """Gets the scopes for extra layers.

  Args:
    last_layers_contain_logits_only: Boolean, True if only consider logits as
    the last layer (i.e., exclude ASPP module, decoder module and so on)

  Returns:
    A list of scopes for extra layers.
  """
  if last_layers_contain_logits_only:
    return [LOGITS_SCOPE_NAME]
  else:
    return [
        LOGITS_SCOPE_NAME,
        IMAGE_POOLING_SCOPE,
        ASPP_SCOPE,
        CONCAT_PROJECTION_SCOPE,
        DECODER_SCOPE,
    ]




def scale_dimension(dim, scale):
  """Scales the input dimension.

  Args:
    dim: Input dimension (a scalar or a scalar Tensor).
    scale: The amount of scaling applied to the input.

  Returns:
    Scaled dimension.
  """
  if isinstance(dim, tf.Tensor):
    return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)


def single_scale_logits(FLAGS,
                       images,
                       seg_map, # [batch_size, H, W, 1], tf.float32
                       car_nums,
                       areas_sqrt_map,
                       idx_xys,
                       bin_vals,
                       outputs_to_indices,
                       model_options,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False,
                       fine_tune_feature_extractor=True, reuse=False,
                       outputs_to_num_classes=None):
  """Gets the logits for multi-scale inputs.

  The returned logits are all downsampled (due to max-pooling layers)
  for both training and evaluation.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    image_pyramid: Input image scales for multi-scale feature extraction.
    weight_decay: The weight decay for model variables.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    fine_tune_feature_extractor: Fine-tune the feature extractor params or not.

  Returns:
    outputs_to_scales_to_logits: A map of maps from output_type (e.g.,
      semantic prediction) to a dictionary of multi-scale logits names to
      logits. For each output_type, the dictionary has keys which
      correspond to the scales and values which correspond to the logits.
      For example, if `scales` equals [1.0, 1.5], then the keys would
      include 'merged_logits', 'logits_1.00' and 'logits_1.50'.

  Raises:
    ValueError: If model_options doesn't specify crop_size and its
      add_image_level_feature = True, since add_image_level_feature requires
      crop_size information.
  """

  # Compute the height, width for the output logits.
  logits_output_stride = (
      model_options.decoder_output_stride or model_options.output_stride)

  logits_height = scale_dimension(
      tf.shape(images)[1],
      1.0 / logits_output_stride)
  logits_width = scale_dimension(
      tf.shape(images)[2],
      1.0 / logits_output_stride)

  # seg_one_hots_N_rescaled = tf.image.resize_nearest_neighbor(seg_one_hots_N,
      # [logits_height, logits_width], align_corners=True)
  # seg_one_hots_N_rescaled = tf.reshape(seg_one_hots_N_flattened, [-1, logits_height, logits_width, 1])

  outputs_to_logits, outputs_to_logits_map, outputs_to_weights_map, outputs_to_areas_N, outputs_to_weightsum_N = _get_logits_mP( # Here we get the regression 'logits' from features!
        FLAGS,
        images,
        seg_map,
        car_nums,
        areas_sqrt_map,
        idx_xys,
        bin_vals,
        outputs_to_indices,
        model_options,
        weight_decay=weight_decay,
        reuse=tf.AUTO_REUSE, # support for auto-reuse if variable exists!
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm,
        fine_tune_feature_extractor=fine_tune_feature_extractor,
        outputs_to_num_classes=outputs_to_num_classes)

  return outputs_to_logits, outputs_to_logits_map, outputs_to_weights_map, outputs_to_areas_N, outputs_to_weightsum_N

def _get_logits_mP(FLAGS,
                images,
                seg_maps, # [batch_size, H, W, 1] float
                car_nums,
                areas_sqrt_map,
                idx_xys,
                bin_vals,
                outputs_to_indices,
                model_options,
                weight_decay=0.0001,
                reuse=None,
                is_training=False,
                fine_tune_batch_norm=False,
                fine_tune_feature_extractor=True,
                outputs_to_num_classes=None):
  """Gets the logits by atrous/image spatial pyramid pooling.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch
    norm parameters or not.

  Returns:
    outputs_to_logits: A map from output_type to logits.
  """
  features_aspp, end_points, features_backbone = extract_features(
      images,
      model_options,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm,
      fine_tune_feature_extractor=fine_tune_feature_extractor) # (1, 34, 85, 256), ..,  (1, 34, 85, 2048)
  print '||||||||||||||| features_aspp ', features_aspp.get_shape()

  if model_options.decoder_output_stride is not None:
    decoder_height = scale_dimension(model_options.crop_size[0],
              1.0 / model_options.decoder_output_stride)
    decoder_width = scale_dimension(model_options.crop_size[1],
              1.0 / model_options.decoder_output_stride)


    features = refine_by_decoder(
        features_aspp,
        end_points,
        decoder_height=decoder_height,
        decoder_width=decoder_width,
        decoder_use_separable_conv=model_options.decoder_use_separable_conv,
        model_variant=model_options.model_variant,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)
    v_coords = tf.range(tf.shape(features)[1])
    u_coords = tf.range(tf.shape(features)[2])
    Vs, Us = tf.meshgrid(v_coords, u_coords)
    features_Ys = tf.tile(tf.expand_dims(tf.expand_dims(tf.transpose(Vs + tf.shape(features)[1]), -1), 0), [tf.shape(features)[0], 1, 1, 1]) # NOTE: added hald height for computing in original size (not halfed)
    features_Xs = tf.tile(tf.expand_dims(tf.expand_dims(tf.transpose(Us), -1), 0), [tf.shape(features)[0], 1, 1, 1])
    print '== Features_aspp/backbone, features', features_aspp.get_shape(), features_backbone.get_shape(), features.get_shape() # (2, 34, 85, 256) (2, 34, 85, 2048) (2, 68, 170, 256) (2, 68, 170, 258)
    features_weight = refine_by_decoder(
        features_aspp,
        end_points,
        decoder_height=decoder_height,
        decoder_width=decoder_width,
        decoder_use_separable_conv=model_options.decoder_use_separable_conv,
        model_variant=model_options.model_variant,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm,
        decoder_scope=WEIGHTS_DECODER_SCOPE)

    if FLAGS.if_zoom:
        zoom_height_start = images.get_shape()[1].value // 4
        zoom_height_end = zoom_height_start * 2
        print zoom_height_start, zoom_height_end, zoom_height_end-zoom_height_start
        images_zoom = tf.slice(images, [0, zoom_height_start, 0, 0],
                [-1, zoom_height_end-zoom_height_start, -1, -1])
                # [-1, 176, -1, -1])

        model_options_zoom = common.ModelOptions(
                outputs_to_num_classes=outputs_to_num_classes,
                crop_size=[images_zoom.get_shape()[1].value, images_zoom.get_shape()[2].value],
                atrous_rates=FLAGS.atrous_rates,
                output_stride=FLAGS.output_stride/2)
        with tf.variable_scope('feature_zoom'):
            features_aspp_zoom, end_points_zoom, _ = extract_features(
              images_zoom,
              model_options_zoom,
              weight_decay=weight_decay,
              reuse=reuse,
              is_training=is_training,
              fine_tune_batch_norm=fine_tune_batch_norm,
              fine_tune_feature_extractor=fine_tune_feature_extractor)
        print '||||||||||||||| features_aspp_zoom ', features_aspp_zoom.get_shape()
        # print end_points_zoom

        decoder_height_zoom = scale_dimension(model_options_zoom.crop_size[0],
                  1.0 / model_options_zoom.decoder_output_stride)
        decoder_width_zoom = scale_dimension(model_options.crop_size[1],
                  1.0 / model_options_zoom.decoder_output_stride)
        features_zoom = refine_by_decoder(
            features_aspp_zoom,
            end_points_zoom,
            decoder_height=decoder_height_zoom,
            decoder_width=decoder_width_zoom,
            decoder_use_separable_conv=model_options.decoder_use_separable_conv,
            model_variant=model_options.model_variant,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm,
            decoder_scope_posefix='-zoom',
            end_points_prefix='feature_zoom')

        print '== Zoom Features_aspp, features:', features_aspp_zoom.get_shape(), features_zoom.get_shape()

        features_weight_zoom = refine_by_decoder(
            features_aspp_zoom,
            end_points_zoom,
            decoder_height=decoder_height_zoom,
            decoder_width=decoder_width_zoom,
            decoder_use_separable_conv=model_options.decoder_use_separable_conv,
            model_variant=model_options.model_variant,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm,
            decoder_scope=WEIGHTS_DECODER_SCOPE,
            decoder_scope_posefix='-zoom',
            end_points_prefix='feature_zoom')

        # features_zoom_padded = tf.concat([tf.zeros_like(features_zoom), features_zoom, tf.zeros_like(features_zoom), tf.zeros_like(features_zoom)], axis=1)
        # features_weight_zoom_padded = tf.concat([tf.zeros_like(features_weight_zoom), features_weight_zoom, tf.zeros_like(features_weight_zoom), tf.zeros_like(features_weight_zoom)], axis=1)

        # features = tf.add(features, features_zoom_padded)
        # features_weight = tf.add(features_weight, features_weight_zoom_padded)

        height_feat = features.get_shape()[1].value
        features = tf.concat([tf.slice(features, [0, 0, 0, 0], [-1, height_feat//4, -1, -1]),
            tf.slice(features, [0, height_feat//4, 0, 0], [-1, height_feat//4, -1, -1])/2. + features_zoom/2.,
            tf.slice(features, [0, height_feat//4*2, 0, 0], [-1, height_feat//4, -1, -1]),
            tf.slice(features, [0, height_feat//4*3, 0, 0], [-1, height_feat//4, -1, -1])], axis=1)
        features_weight = tf.concat([tf.slice(features_weight, [0, 0, 0, 0], [-1, height_feat//4, -1, -1]),
            tf.slice(features_weight, [0, height_feat//4, 0, 0], [-1, height_feat//4, -1, -1])/2. + features_weight_zoom/2.,
            tf.slice(features_weight, [0, height_feat//4*2, 0, 0], [-1, height_feat//4, -1, -1]),
            tf.slice(features_weight, [0, height_feat//4*3, 0, 0], [-1, height_feat//4, -1, -1])], axis=1)

  features_concat = tf.concat([features,
        tf.to_float(features_Xs * model_options.decoder_output_stride),
        tf.to_float(features_Ys) * model_options.decoder_output_stride,
        areas_sqrt_map], axis=3)
  features_weight_concat = tf.concat([features_weight,
      tf.to_float(features_Xs) * model_options.decoder_output_stride,
      tf.to_float(features_Ys) * model_options.decoder_output_stride,
      areas_sqrt_map], axis=3)

  outputs_to_logits_N = {}
  outputs_to_logits_map = {}
  outputs_to_weights_map = {}
  outputs_to_areas_N = {}
  outputs_to_weightsum_N = {}

  N_batch_idxs = tf.reshape(tf.slice(idx_xys, [0, 0], [-1, 1]), [-1])
  # last_dim = tf.shape(features)[3]

  for output in sorted(model_options.outputs_to_num_classes):
    last_dim = model_options.outputs_to_num_classes[output]

    weights = get_branch_logits(
      features_weight_concat,
      1,
      model_options.atrous_rates,
      aspp_with_batch_norm=model_options.aspp_with_batch_norm,
      kernel_size=model_options.logits_kernel_size,
      weight_decay=weight_decay,
      reuse=reuse,
      scope_suffix=output+'_weights',
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm,
      if_bn = False) # (batch_size, 68, 170, 1)

    logits = get_branch_logits(
        features_concat,
        model_options.outputs_to_num_classes[output],
        model_options.atrous_rates,
        aspp_with_batch_norm=model_options.aspp_with_batch_norm,
        kernel_size=model_options.logits_kernel_size,
        weight_decay=weight_decay,
        reuse=reuse,
        scope_suffix=output+'_logits',
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm,
        if_bn =False,
        activation=None)

    if output == 'x' and FLAGS.if_uvflow:
        logits = logits + tf.to_float(features_Xs) * model_options.decoder_output_stride
        print '||||||||added grids to x'
    if output == 'y' and FLAGS.if_uvflow:
        logits = logits + tf.to_float(features_Ys) * model_options.decoder_output_stride
        print '||||||||added grids to y'

    outputs_to_logits_map[output] = logits

    # areas_sqrt_N = get_areas_N(seg_maps, car_nums)
    # print areas_N.get_shape()
    # programPause = raw_input("Press the <ENTER> key to continue...")

    # Working
    with tf.variable_scope('per_car_logits_aggre_2'):
        num_samples = tf.shape(logits)[0]
        init_array = tf.TensorArray(tf.float32, size=num_samples, infer_shape=False) # https://stackoverflow.com/questions/43270849/tensorflow-map-fn-tensorarray-has-inconsistent-shapes
        def loop_body_per_sample(i, ta):
            logits_sample = tf.gather(logits, i) # (68, 170, 32)
            weight_sample = tf.gather(weights, i) # (68, 170, 1), in [0., 1.]
            seg_map_sample = tf.gather(seg_maps, i) # (68, 170, 1)
            car_num_sample = tf.gather(car_nums, i) # ()

            seg_one_hot_sample = tf.one_hot(tf.cast(tf.squeeze(seg_map_sample), tf.int32), depth=car_num_sample+1) # (68, 170, ?+1)
            seg_one_hot_sample = tf.transpose(tf.slice(seg_one_hot_sample, [0, 0, 1], [-1, -1, -1]), [2, 0, 1]) # (?, 68, 170)

            def per_car(seg_one_hot_car):
                seg_one_hot_car_bool = tf.cast(seg_one_hot_car, tf.bool)
                logits_car_masked = tf.boolean_mask(logits_sample, seg_one_hot_car_bool)
                weight_car_masked = tf.boolean_mask(weight_sample, seg_one_hot_car_bool) # [-1, 1]

                # weight_sum = tf.reduce_sum(weight_car_masked)+1e-10
                # weight_car_normlized = weight_car_masked / weight_sum

                weight_car_masked_ones = tf.ones_like(weight_car_masked)
                weight_sum_ones = tf.reduce_sum(weight_car_masked_ones)+1e-10
                weight_car_normlized_ones = weight_car_masked_ones / weight_sum_ones

                weight_car_masked_sum = tf.reduce_sum(tf.exp(weight_car_masked)) + 1e-10
                weight_car_normlized = tf.exp(weight_car_masked) / weight_car_masked_sum

                area_car = tf.to_float(tf.reduce_sum(seg_one_hot_car))

                logits_car_weighted = tf.multiply(logits_car_masked, weight_car_normlized) # [-1, 256]
                logits_car_aggre = tf.reshape(tf.reduce_sum(logits_car_weighted, 0), [-1]) # (256,)
                return tf.cond(tf.equal(tf.size(weight_car_masked), 0),
                        lambda: tf.zeros([tf.shape(logits)[-1]+2], dtype=tf.float32),
                        lambda: tf.concat([logits_car_aggre, tf.reshape(area_car, [-1]), tf.reshape(weight_car_masked_sum/(area_car+1e-10), [-1])], axis=0))
            logits_area_weightsum_sample = tf.map_fn(per_car, seg_one_hot_sample, dtype=tf.float32) # [?, 256+1]
            logits_sample = tf.slice(logits_area_weightsum_sample, [0, 0], [-1, last_dim]) # [?, 256]
            areas_sample = tf.slice(logits_area_weightsum_sample, [0, last_dim], [-1, 1]) # [?, 1]
            weightsum_sample = tf.slice(logits_area_weightsum_sample, [0, last_dim+1], [-1, 1]) # [?, 1]
            logits_areas_weightsum_sample = tf.concat([logits_sample, areas_sample, weightsum_sample], axis=-1)
            logits_areas_weightsum_sample.set_shape([None, last_dim+2])

            return i + 1, ta.write(i, logits_areas_weightsum_sample)

        _, logits_areas_all = tf.while_loop(lambda i, ta: i < num_samples, loop_body_per_sample, [0, init_array])

    logits_areas_N = tf.reshape(logits_areas_all.concat(), [-1, last_dim+2])
    logits_N = tf.slice(logits_areas_N, [0, 0], [-1, last_dim])
    areas_N = tf.slice(logits_areas_N, [0, last_dim], [-1, 1])
    weightsum_N = tf.slice(logits_areas_N, [0, last_dim+1], [-1, 1])

    outputs_to_logits_N[output] = logits_N # [N, 32]
    print logits.get_shape(), features_Xs.get_shape(), logits.get_shape(), outputs_to_logits_N[output].get_shape(), '+++++++++++++', output # (2, 68, 170, 64) (2, 68, 170, 1) (2, 68, 170, 64) (?, 64)
    outputs_to_weights_map[output] = weights # [batch_size, H', W', 1]
    outputs_to_areas_N[output] = areas_N # [N, 1]
    outputs_to_weightsum_N[output] = weightsum_N # [N, 1]

  return outputs_to_logits_N, outputs_to_logits_map, outputs_to_weights_map, outputs_to_areas_N, outputs_to_weightsum_N

def get_areas_N(seg_maps, car_nums):
    with tf.variable_scope('per_car_area'):
        num_samples = tf.shape(seg_maps)[0]
        init_array = tf.TensorArray(tf.float32, size=num_samples, infer_shape=False) # https://stackoverflow.com/questions/43270849/tensorflow-map-fn-tensorarray-has-inconsistent-shapes
        def loop_body_per_sample(i, ta):
            seg_map_sample = tf.gather(seg_maps, i) # (68, 170, 1)
            car_num_sample = tf.gather(car_nums, i) # ()
            seg_one_hot_sample = tf.one_hot(tf.cast(tf.squeeze(seg_map_sample), tf.int32), depth=car_num_sample+1) # (68, 170, ?+1)
            seg_one_hot_sample = tf.transpose(tf.slice(seg_one_hot_sample, [0, 0, 1], [-1, -1, -1]), [2, 0, 1]) # (?, 68, 170)

            def per_car(seg_one_hot_car):
                seg_one_hot_car_bool = tf.cast(seg_one_hot_car, tf.bool)
                area_car = tf.to_float(tf.reduce_sum(seg_one_hot_car))
                return tf.cond(tf.equal(tf.size(area_car), 0),
                        lambda: tf.zeros([1], dtype=tf.float32),
                        lambda: tf.reshape(tf.sqrt(area_car), [-1]))
            areas_sample = tf.map_fn(per_car, seg_one_hot_sample, dtype=tf.float32) # [?, 1]

            return i + 1, ta.write(i, areas_sample)

        _, areas_all = tf.while_loop(lambda i, ta: i < num_samples, loop_body_per_sample, [0, init_array])

    areas_N = tf.reshape(areas_all.concat(), [-1, 1])
    return areas_N

def extract_features(images,
                     model_options,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False,
                     fine_tune_feature_extractor=True,
                     output_stride=None):
  """Extracts features by the particular model_variant.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    concat_logits: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined by
      the images height/width and output_stride.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  """
  features, end_points = feature_extractor.extract_features(
      images,
      output_stride=model_options.output_stride if output_stride is None else output_stride,
      multi_grid=model_options.multi_grid,
      model_variant=model_options.model_variant,
      depth_multiplier=model_options.depth_multiplier,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm)

  if not model_options.aspp_with_batch_norm:
    return features, end_points
  else:
    return aspp_with_batch_norm(features, model_options, weight_decay, reuse, is_training, fine_tune_batch_norm, activation_fn=tf.nn.relu, depth=256), end_points, features

def aspp_with_batch_norm(features,
                         model_options,
                         weight_decay=0.0001,
                         reuse=None,
                         is_training=False,
                         fine_tune_batch_norm=False,
                         fine_tune_feature_extractor=True,
                         activation_fn=tf.nn.relu,
                         depth=256):
    batch_norm_params = {
        'is_training': is_training and fine_tune_batch_norm,
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True,
    }

    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=activation_fn,
        normalizer_fn=slim.batch_norm,
        padding='SAME',
        stride=1,
        reuse=reuse):
      with slim.arg_scope([slim.batch_norm], **batch_norm_params):
        depth = depth
        branch_logits = []

        if model_options.add_image_level_feature:
          pool_height = scale_dimension(model_options.crop_size[0],
                                        1. / model_options.output_stride)
          pool_width = scale_dimension(model_options.crop_size[1],
                                       1. / model_options.output_stride)
          image_feature = slim.avg_pool2d(
              features, [pool_height, pool_width], [pool_height, pool_width],
              padding='VALID')
          image_feature = slim.conv2d(
              image_feature, depth, 1, scope=IMAGE_POOLING_SCOPE)
          image_feature = tf.image.resize_bilinear(
              image_feature, [pool_height, pool_width], align_corners=True)
          image_feature.set_shape([None, pool_height, pool_width, depth])
          branch_logits.append(image_feature)

        # Employ a 1x1 convolution.
        branch_logits.append(slim.conv2d(features, depth, 1,
                                         scope=ASPP_SCOPE + str(0)))

        if model_options.atrous_rates:
          # Employ 3x3 convolutions with different atrous rates.
          for i, rate in enumerate(model_options.atrous_rates, 1):
            scope = ASPP_SCOPE + str(i)
            if model_options.aspp_with_separable_conv:
              aspp_features = split_separable_conv2d(
                  features,
                  filters=depth,
                  rate=rate,
                  weight_decay=weight_decay,
                  scope=scope)
            else:
              aspp_features = slim.conv2d(
                  features, depth, 3, rate=rate, scope=scope)
            branch_logits.append(aspp_features)

        # Merge branch logits.
        concat_logits = tf.concat(branch_logits, 3)
        concat_logits = slim.conv2d(
            concat_logits, depth, 1, scope=CONCAT_PROJECTION_SCOPE)
        concat_logits = slim.dropout(
            concat_logits,
            keep_prob=0.9,
            is_training=is_training,
            scope=CONCAT_PROJECTION_SCOPE + '_dropout')
        return concat_logits

def refine_by_decoder(features,
                      end_points,
                      decoder_height,
                      decoder_width,
                      decoder_use_separable_conv=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False,
                      activation_fn=tf.nn.relu,
                      decoder_scope=DECODER_SCOPE,
                      weights_initializer=None,
                      decoder_depth=256,
                      decoder_scope_posefix='',
                      end_points_prefix=None):
  """Adds the decoder to obtain sharper segmentation results.

  Args:
    features: A tensor of size [batch, features_height, features_width,
      features_channels].
    end_points: A dictionary from components of the network to the corresponding
      activation.
    decoder_height: The height of decoder feature maps.
    decoder_width: The width of decoder feature maps.
    decoder_use_separable_conv: Employ separable convolution for decoder or not.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

  Returns:
    Decoder output with size [batch, decoder_height, decoder_width,
      decoder_channels].
  """
  batch_norm_params = {
      'is_training': is_training and fine_tune_batch_norm,
      'decay': 0.9997,
      'epsilon': 1e-5,
      'scale': True,
  }

  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=weights_initializer,
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm,
      padding='SAME',
      stride=1,
      reuse=reuse):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with tf.variable_scope(decoder_scope+decoder_scope_posefix, decoder_scope, [features]):
        feature_list = feature_extractor.networks_to_feature_maps[
            model_variant][feature_extractor.DECODER_END_POINTS]
        if feature_list is None:
          tf.logging.info('Not found any decoder end points.')
          return features
        else:
          decoder_features = features
          for i, name in enumerate(feature_list):
            decoder_features_list = [decoder_features]

            # MobileNet variants use different naming convention.
            if 'mobilenet' in model_variant:
              feature_name = name
            else:
              feature_name = '{}/{}'.format(
                  feature_extractor.name_scope[model_variant], name)
            decoder_features_list.append(
                slim.conv2d(
                    end_points[feature_name if end_points_prefix is None else '%s/%s'%(end_points_prefix, feature_name)],
                    48,
                    1,
                    scope='feature_projection' + str(i)))
            # Resize to decoder_height/decoder_width.
            for j, feature in enumerate(decoder_features_list):
              decoder_features_list[j] = tf.image.resize_bilinear(
                  feature, [decoder_height, decoder_width], align_corners=True)
              decoder_features_list[j].set_shape(
                  [None, decoder_height, decoder_width, None])
            # decoder_depth = 256
            if decoder_use_separable_conv:
              decoder_features = split_separable_conv2d(
                  tf.concat(decoder_features_list, 3),
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv0')
              decoder_features = split_separable_conv2d(
                  decoder_features,
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv1')
            else:
              num_convs = 2
              decoder_features = slim.repeat(
                  tf.concat(decoder_features_list, 3),
                  num_convs,
                  slim.conv2d,
                  decoder_depth,
                  3,
                  scope='decoder_conv' + str(i))
          return decoder_features


def get_branch_logits(features,
                      num_classes,
                      atrous_rates=None,
                      aspp_with_batch_norm=False,
                      kernel_size=1,
                      weight_decay=0.0001,
                      reuse=None,
                      scope_suffix='',
                      is_training=False,
                      fine_tune_batch_norm=False,
                      if_bn=False,
                      activation=None):
  """Gets the logits from each model's branch.

  The underlying model is branched out in the last layer when atrous
  spatial pyramid pooling is employed, and all branches are sum-merged
  to form the final logits.

  Args:
    features: A float tensor of shape [batch, height, width, channels].
    num_classes: Number of classes to predict.
    atrous_rates: A list of atrous convolution rates for last layer.
    aspp_with_batch_norm: Use batch normalization layers for ASPP.
    kernel_size: Kernel size for convolution.
    weight_decay: Weight decay for the model variables.
    reuse: Reuse model variables or not.
    scope_suffix: Scope suffix for the model variables.

  Returns:
    Merged logits with shape [batch, height, width, num_classes].

  Raises:
    ValueError: Upon invalid input kernel_size value.
  """
  # When using batch normalization with ASPP, ASPP has been applied before
  # in extract_features, and thus we simply apply 1x1 convolution here.
  if aspp_with_batch_norm or atrous_rates is None:
    if kernel_size != 1:
      raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                       'using aspp_with_batch_norm. Gets %d.' % kernel_size)
    atrous_rates = [1]

  batch_norm_params = {
    'is_training': is_training,
    # 'decay': 0.9997,
    # 'epsilon': 1e-5,
    # 'scale': True,
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      # normalizer_fn=slim.batch_norm if if_bn else None,
      reuse=reuse):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
        with tf.variable_scope(LOGITS_SCOPE_NAME, LOGITS_SCOPE_NAME, [features]):
          branch_logits = []
          for i, rate in enumerate(atrous_rates):
            scope = scope_suffix
            if i:
              scope += '_%d' % i

            branch_logits.append(
                slim.conv2d(
                    features,
                    num_classes,
                    kernel_size=kernel_size,
                    rate=rate,
                    activation_fn=activation,
                    scope=scope))

          return tf.add_n(branch_logits)


def not_predict_labels_multi_scale(images,
                               model_options,
                               eval_scales=(1.0,),
                               add_flipped_images=False):
  """Predicts segmentation labels.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    eval_scales: The scales to resize images for evaluation.
    add_flipped_images: Add flipped images for evaluation or not.

  Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
      prediction) and values storing Tensors representing predictions (argmax
      over channels). Each prediction has size [batch, height, width].
  """
  outputs_to_predictions = {
      output: []
      for output in model_options.outputs_to_num_classes
  }

  for i, image_scale in enumerate(eval_scales):
    with tf.variable_scope(tf.get_variable_scope(), reuse=True if i else None):
      outputs_to_scales_to_logits = multi_scale_logits(
          images,
          model_options=model_options,
          image_pyramid=[image_scale],
          is_training=False,
          fine_tune_batch_norm=False)

    if add_flipped_images:
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        outputs_to_scales_to_logits_reversed = multi_scale_logits(
            tf.reverse_v2(images, [2]),
            model_options=model_options,
            image_pyramid=[image_scale],
            is_training=False,
            fine_tune_batch_norm=False)

    for output in sorted(outputs_to_scales_to_logits):
      scales_to_logits = outputs_to_scales_to_logits[output]
      logits = tf.image.resize_bilinear(
          scales_to_logits[MERGED_LOGITS_SCOPE],
          tf.shape(images)[1:3],
          align_corners=True)
      outputs_to_predictions[output].append(
          tf.expand_dims(tf.nn.softmax(logits), 4))

      if add_flipped_images:
        scales_to_logits_reversed = (
            outputs_to_scales_to_logits_reversed[output])
        logits_reversed = tf.image.resize_bilinear(
            tf.reverse_v2(scales_to_logits_reversed[MERGED_LOGITS_SCOPE], [2]),
            tf.shape(images)[1:3],
            align_corners=True)
        outputs_to_predictions[output].append(
            tf.expand_dims(tf.nn.softmax(logits_reversed), 4))

  for output in sorted(outputs_to_predictions):
    predictions = outputs_to_predictions[output]
    # Compute average prediction across different scales and flipped images.
    predictions = tf.reduce_mean(tf.concat(predictions, 4), axis=4)
    outputs_to_predictions[output] = tf.argmax(predictions, 3)

  return outputs_to_predictions


def predict_labels(images, model_options, image_pyramid=None):
  """Predicts segmentation labels.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    image_pyramid: Input image scales for multi-scale feature extraction.

  Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
      prediction) and values storing Tensors representing predictions (argmax
      over channels). Each prediction has size [batch, height, width].
  """
  outputs_to_scales_to_logits = multi_scale_logits(
      images,
      model_options=model_options,
      image_pyramid=image_pyramid,
      is_training=False,
      fine_tune_batch_norm=False)

  predictions = {}
  for output in sorted(outputs_to_scales_to_logits):
    scales_to_logits = outputs_to_scales_to_logits[output]
    logits = tf.image.resize_bilinear(
        scales_to_logits[MERGED_LOGITS_SCOPE],
        tf.shape(images)[1:3],
        align_corners=True)
    predictions[output] = tf.argmax(logits, 3)

  return predictions

def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):
  """Splits a separable conv2d into depthwise and pointwise conv2d.

  This operation differs from `tf.layers.separable_conv2d` as this operation
  applies activation function between depthwise and pointwise conv2d.

  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.

  Returns:
    Computed features after split separable conv2d.
  """
  outputs = slim.separable_conv2d(
      inputs,
      None,
      kernel_size=kernel_size,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')
