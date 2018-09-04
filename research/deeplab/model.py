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

slim = tf.contrib.slim

LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'


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


def single_scale_logits(images,
                       seg_map, # [batch_size, H, W, 1], tf.float32
                       car_nums,
                       idx_xys,
                       model_options,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False,
                       fine_tune_feature_extractor=True, reuse=False):
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


  outputs_to_logits, outputs_to_weights = _get_logits_mP( # Here we get the regression 'logits' from features!
        images,
        seg_map,
        car_nums,
        idx_xys,
        model_options,
        weight_decay=weight_decay,
        reuse=tf.AUTO_REUSE, # support for auto-reuse if variable exists!
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm,
        fine_tune_feature_extractor=fine_tune_feature_extractor)

  return outputs_to_logits, outputs_to_weights



def _get_logits_mP(images,
                seg_maps, # [batch_size, H, W, 1] float
                car_nums,
                idx_xys,
                model_options,
                weight_decay=0.0001,
                reuse=None,
                is_training=False,
                fine_tune_batch_norm=False,
                fine_tune_feature_extractor=True):
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
  features, end_points = extract_features(
      images,
      model_options,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm,
      fine_tune_feature_extractor=fine_tune_feature_extractor) # [batch_size, 68, 170, 256]

  if model_options.decoder_output_stride is not None:
    decoder_height = scale_dimension(model_options.crop_size[0],
                                     1.0 / model_options.decoder_output_stride)
    decoder_width = scale_dimension(model_options.crop_size[1],
                                    1.0 / model_options.decoder_output_stride)
    features = refine_by_decoder(
        features,
        end_points,
        decoder_height=decoder_height,
        decoder_width=decoder_width,
        decoder_use_separable_conv=model_options.decoder_use_separable_conv,
        model_variant=model_options.model_variant,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

  outputs_to_logits = {}
  outputs_to_weights = {}

  N_batch_idxs = tf.reshape(tf.slice(idx_xys, [0, 0], [-1, 1]), [-1])
  print car_nums.get_shape(), car_nums.dtype

  for output in sorted(model_options.outputs_to_num_classes):
    weights = (get_branch_logits(
      features,
      1,
      model_options.atrous_rates,
      aspp_with_batch_norm=model_options.aspp_with_batch_norm,
      kernel_size=model_options.logits_kernel_size,
      weight_decay=weight_decay,
      reuse=reuse,
      scope_suffix=output,
      activation=tf.tanh,
      normalizer_fn=slim.batch_norm) + 1.) / 2. # (batch_size, 68, 170, 1)

    # Too large to fit into GPU
    # features_N = tf.gather(features, N_batch_idxs) # [N, 68, 170, 256]
    # weights_repeatN = tf.gather(weights, N_batch_idxs) # [N, 68, 170, 1]
    # weights_N = tf.multiply(weights_repeatN, seg_one_hots_N_rescaled) # [N, 68, 170, 1]
    # weights_N_sums = tf.reduce_sum(tf.reduce_sum(weights_N, axis=1, keepdims=True), axis=2, keepdims=True)+1e-6 # [N, 1, 1, 1]
    # weights_N_normalized = tf.multiply(weights_N, tf.reciprocal(weights_N_sums))
    # features_N_weighted = tf.multiply(features_N, weights_N_normalized) # [N, 68, 170, 256]
    # features_N_aggre = tf.reduce_sum(tf.reduce_sum(features_N_weighted, 1, keepdims=True), 2, keepdims=True) # (N, 1, 1, 256)

    # Worked
    # with tf.variable_scope('per_car_feature_aggre'):
    #     def per_car(inputs):
    #         batch_id = tf.gather(inputs[0], 0)
    #         seg_one_hots_car = tf.squeeze(inputs[1])
    #         # features_car = tf.gather(features, batch_id) # [68, 170, 256]
    #         # features_car = tf.squeeze(tf.slice(features, [batch_id, 0, 0, 0], [1, -1, -1, -1]), 0) # [68, 170, 256]
    #         features_car_masked = tf.boolean_mask(
    #                 tf.squeeze(tf.slice(features, [batch_id, 0, 0, 0], [1, -1, -1, -1]), 0),
    #                 tf.cast(seg_one_hots_car, tf.bool))# [68, 170, 256]
    #         # print '----', features_car_masked.get_shape()
    #         # weights_car = tf.gather(weights, batch_id) # [68, 170, 1]
    #         # weights_car = tf.squeeze(tf.slice(weights, [batch_id, 0, 0, 0], [1, -1, -1, -1]), 0) # [68, 170, 1]
    #         weights_car_masked = tf.boolean_mask(
    #                 tf.squeeze(tf.slice(weights, [batch_id, 0, 0, 0], [1, -1, -1, -1]), 0),
    #                 tf.cast(seg_one_hots_car, tf.bool))# [68, 170, 256]
    #         # weights_car_masked = tf.multiply(weights_car, tf.to_float(seg_one_hots_car)) # [68, 170, 1]
    #         weights_car_normlized = weights_car_masked / (tf.reduce_sum(weights_car_masked)+1e-6)
    #         features_car_weighted = tf.multiply(features_car_masked, weights_car_normlized) # [68, 170, 256]
    #         # features_car_aggre = tf.reduce_sum(tf.reduce_sum(features_car_weighted, 0, keepdims=True), 1, keepdims=True) # (1, 1, 256)
    #         features_car_aggre = tf.expand_dims(tf.reduce_sum(features_car_weighted, 0, keepdims=True), 0) # (1, 1, 256)
    #         return features_car_aggre
    #     features_N_aggre = tf.map_fn(per_car, (idx_xys, seg_one_hots_N_rescaled), dtype=tf.float32) # [N, 1, 1, 256]

    # with slim.arg_scope(
    #     [slim.conv2d],
    #     weights_regularizer=slim.l2_regularizer(weight_decay),
    #     weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
    #     reuse=reuse):
    #   with tf.variable_scope(LOGITS_SCOPE_NAME, LOGITS_SCOPE_NAME, [features_N_aggre]):
    #     logits_car = slim.conv2d(
    #               features_N_aggre,
    #               model_options.outputs_to_num_classes[output],
    #               kernel_size=1,
    #               activation_fn=None,
    #               normalizer_fn=None,
    #               scope='feature_reg_car')

    # outputs_to_logits[output] = tf.squeeze(tf.squeeze(logits_car, 1), 1) # (num_cars, 32)

    # Working
    with tf.variable_scope('per_car_feature_aggre_2'):
        num_samples = tf.shape(features)[0]
        init_array = tf.TensorArray(tf.float32, size=num_samples, infer_shape=False) # https://stackoverflow.com/questions/43270849/tensorflow-map-fn-tensorarray-has-inconsistent-shapes
        def loop_body_per_sample(i, ta):
            feature_sample = tf.gather(features, i) # (68, 170, 256)
            weight_sample = tf.gather(weights, i) # (68, 170, 1), in [0., 1.]
            seg_map_sample = tf.gather(seg_maps, i) # (68, 170, 1)
            car_num_sample = tf.gather(car_nums, i) # ()

            seg_one_hot_sample = tf.one_hot(tf.cast(tf.squeeze(seg_map_sample), tf.int32), depth=car_num_sample+1) # (68, 170, ?+1)
            seg_one_hot_sample = tf.transpose(tf.slice(seg_one_hot_sample, [0, 0, 1], [-1, -1, -1]), [2, 0, 1]) # (?, 68, 170)

            def per_car(seg_one_hot_car):
                seg_one_hot_car_bool = tf.cast(seg_one_hot_car, tf.bool)
                feature_car_masked = tf.boolean_mask(feature_sample, seg_one_hot_car_bool)
                weight_car_masked = tf.boolean_mask(weight_sample, seg_one_hot_car_bool)
                weight_car_normlized = weight_car_masked / (tf.reduce_sum(weight_car_masked)+1e-6)
                feature_car_weighted = tf.multiply(feature_car_masked, weight_car_normlized) # [-1, 256]
                feature_car_aggre = tf.reduce_sum(feature_car_weighted, 0) # (256,)
                return tf.cond(tf.rank(feature_car_masked)<2,
                        lambda: tf.zeros([tf.shape(features)[-1]], dtype=tf.float32), lambda: feature_car_aggre)
            features_sample = tf.map_fn(per_car, seg_one_hot_sample, dtype=tf.float32) # [?, 256]

            return i + 1, ta.write(i, features_sample)

        _, features_all = tf.while_loop(lambda i, ta: i < num_samples, loop_body_per_sample, [0, init_array])

        # def per_sample(inputs):
        #     feature_sample = inputs[0] # (68, 170, 256)
        #     weight_sample = inputs[1] # (68, 170, 1)
        #     seg_map_sample = inputs[2] # (68, 170, 1)
        #     car_num_sample = inputs[3] # ()

        #     seg_one_hot_sample = tf.one_hot(tf.cast(tf.squeeze(seg_map_sample), tf.int32), depth=car_num_sample+1) # (68, 170, ?+1)
        #     seg_one_hot_sample = tf.transpose(tf.slice(seg_one_hot_sample, [0, 0, 1], [-1, -1, -1]), [2, 0, 1]) # (?, 68, 170)
        #     # print '+++', feature_sample.get_shape(), weight_sample.get_shape(), seg_map_sample.get_shape(), car_num_sample.get_shape()
        #     # print seg_one_hot_sample.get_shape()

        #     def per_car(seg_one_hot_car):
        #         seg_one_hot_car_bool = tf.cast(seg_one_hot_car, tf.bool)
        #         feature_car_masked = tf.boolean_mask(feature_sample, seg_one_hot_car_bool)
        #         weight_car_masked = tf.boolean_mask(weight_sample, seg_one_hot_car_bool)
        #         weight_car_normlized = weight_car_masked / (tf.reduce_sum(weight_car_masked)+1e-6)
        #         feature_car_weighted = tf.multiply(feature_car_masked, weight_car_normlized) # [-1, 256]
        #         feature_car_aggre = tf.reduce_sum(feature_car_weighted, 0) # (1, 1, 256)
        #         return feature_car_aggre
        #     features_sample = tf.map_fn(per_car, seg_one_hot_sample, dtype=tf.float32) # [?, 32]
        #     return features_sample

        # features_all = tf.map_fn(per_sample, (features, weights, seg_maps, car_nums), dtype=tf.float32, infer_shape=False) # [N, ?, 256]

    # last_dim = 256 if 'mobilenet' in model_options.model_variant else 2048
    last_dim = tf.shape(features)[3]
    # features_N = tf.reshape(tf.gather_nd(features_all.stack(), idx_xys), [-1, last_dim])
    features_N = tf.reshape(features_all.concat(), [-1, last_dim])
    print features_N.get_shape()
    features_N = tf.expand_dims(tf.expand_dims(features_N, 1), 1)

    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
        reuse=reuse):
      # print features_N.get_shape(), '++++++++++++++', model_options.outputs_to_num_classes[output]
      with tf.variable_scope('FEATURE_REG'):
        logits_N = slim.conv2d(
                  features_N,
                  model_options.outputs_to_num_classes[output],
                  kernel_size=1,
                  activation_fn=None,
                  normalizer_fn=None,
                  scope='feature_reg_car-%s'%output)
    print '|||||', logits_N.get_shape()
    outputs_to_logits[output] = tf.squeeze(tf.squeeze(logits_N, 1), 1)
    outputs_to_weights[output] = weights
    print '=========', outputs_to_logits[output].get_shape()
  return outputs_to_logits, outputs_to_weights

# def _get_logits(images,
#                 seg_int,
#                 model_options,
#                 weight_decay=0.0001,
#                 reuse=None,
#                 is_training=False,
#                 fine_tune_batch_norm=False,
#                 fine_tune_feature_extractor=True):
#   """Gets the logits by atrous/image spatial pyramid pooling.

#   Args:
#     images: A tensor of size [batch, height, width, channels].
#     model_options: A ModelOptions instance to configure models.
#     weight_decay: The weight decay for model variables.
#     reuse: Reuse the model variables or not.
#     is_training: Is training or not.
#     fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

#   Returns:
#     outputs_to_logits: A map from output_type to logits.
#   """
#   features, end_points = extract_features(
#       images,
#       model_options,
#       weight_decay=weight_decay,
#       reuse=reuse,
#       is_training=is_training,
#       fine_tune_batch_norm=fine_tune_batch_norm,
#       fine_tune_feature_extractor=fine_tune_feature_extractor) # (3, 68, 170, 256)

#   if model_options.decoder_output_stride is not None:
#     decoder_height = scale_dimension(model_options.crop_size[0],
#                                      1.0 / model_options.decoder_output_stride)
#     decoder_width = scale_dimension(model_options.crop_size[1],
#                                     1.0 / model_options.decoder_output_stride)
#     features = refine_by_decoder(
#         features,
#         end_points,
#         decoder_height=decoder_height,
#         decoder_width=decoder_width,
#         decoder_use_separable_conv=model_options.decoder_use_separable_conv,
#         model_variant=model_options.model_variant,
#         weight_decay=weight_decay,
#         reuse=reuse,
#         is_training=is_training,
#         fine_tune_batch_norm=fine_tune_batch_norm)

#   outputs_to_logits = {}
#   for output in sorted(model_options.outputs_to_num_classes):
#     outputs_to_logits[output] = get_branch_logits(
#         features,
#         model_options.outputs_to_num_classes[output],
#         model_options.atrous_rates,
#         aspp_with_batch_norm=model_options.aspp_with_batch_norm,
#         kernel_size=model_options.logits_kernel_size,
#         weight_decay=weight_decay,
#         reuse=reuse,
#         scope_suffix=output)

#   return outputs_to_logits


def extract_features(images,
                     model_options,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False,
                     fine_tune_feature_extractor=True):
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
      output_stride=model_options.output_stride,
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
    batch_norm_params = {
        'is_training': is_training and fine_tune_batch_norm,
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True,
    }

    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        padding='SAME',
        stride=1,
        reuse=reuse):
      with slim.arg_scope([slim.batch_norm], **batch_norm_params):
        depth = 256
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

        return concat_logits, end_points


def refine_by_decoder(features,
                      end_points,
                      decoder_height,
                      decoder_width,
                      decoder_use_separable_conv=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False):
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
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      padding='SAME',
      stride=1,
      reuse=reuse):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with tf.variable_scope(DECODER_SCOPE, DECODER_SCOPE, [features]):
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
                    end_points[feature_name],
                    48,
                    1,
                    scope='feature_projection' + str(i)))
            # Resize to decoder_height/decoder_width.
            for j, feature in enumerate(decoder_features_list):
              decoder_features_list[j] = tf.image.resize_bilinear(
                  feature, [decoder_height, decoder_width], align_corners=True)
              decoder_features_list[j].set_shape(
                  [None, decoder_height, decoder_width, None])
            decoder_depth = 256
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
                      activation=None,
                      normalizer_fn=None):
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

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      reuse=reuse):
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
                normalizer_fn=normalizer_fn,
                scope=scope))

      return tf.add_n(branch_logits)


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
