import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.utils import train_utils
import numpy as np

def _build_deeplab(FLAGS, samples, outputs_to_num_classes, outputs_to_indices, bin_vals, dataset, is_training=True, reuse=False):
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
  # samples = inputs_queue.dequeue()

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
  samples['pose_map'] = tf.identity(
      samples['pose_map'], name=is_training_prefix+'pose_map')
  samples['shape_map'] = tf.identity(samples['shape_map'], name=is_training_prefix+'shape_map')
  samples['shape_id_map'] = tf.identity(samples['shape_id_map'], name=is_training_prefix+'shape_id_map')
  samples['shape_id_map_gt'] = tf.identity(samples['shape_id_map_gt'], name=is_training_prefix+'shape_id_map_gt')
  samples['seg'] = tf.identity(samples['seg'], name=is_training_prefix+'seg')

  masks = tf.identity(samples['mask'], name=is_training_prefix+'not_ignore_mask_in_loss')
  count_valid = tf.reduce_sum(tf.to_float(masks))+1e-6

  model_options = common.ModelOptions(
      outputs_to_num_classes=outputs_to_num_classes,
      crop_size=[dataset.height, dataset.width],
      atrous_rates=FLAGS.atrous_rates,
      output_stride=FLAGS.output_stride)

  outputs_to_logits = model.single_scale_logits(
      samples[common.IMAGE],
      model_options=model_options,
      weight_decay=FLAGS.weight_decay,
      is_training=is_training,
      fine_tune_batch_norm=FLAGS.fine_tune_batch_norm and is_training,
      fine_tune_feature_extractor=FLAGS.fine_tune_feature_extractor and is_training)

  # Get regressed logits for all outputs
  scaled_logits_list = []
  reg_logits_list = []
  for output in dataset.output_names:
      if FLAGS.if_discrete_loss:
          prob_logits = train_utils.logits_cls_to_logits_probReg(
                  outputs_to_logits[output],
                  bin_vals[outputs_to_indices[output]])
          reg_logits = prob_logits
      else:
          reg_logits = outputs_to_logits[output]
      reg_logits_list.append(reg_logits)

  ## Regression loss for pose
  balance_rot_reg_loss = 10.
  balance_trans_reg_loss = 1.
  reg_logits_concat = tf.concat(reg_logits_list, axis=3)
  _, scaled_logits_pose, rot_q_error_map, trans_error_map = train_utils.add_my_pose_loss(
          tf.gather(reg_logits_concat, [0, 1, 2, 3, 4, 5, 6], axis=3),
          samples['pose_map'],
          samples['mask'],
          balance_rot=balance_rot_reg_loss,
          balance_trans=balance_trans_reg_loss,
          upsample_logits=FLAGS.upsample_logits,
          name=is_training_prefix + 'loss_reg',
          loss_collection=None if is_training else None
          )
  scaled_logits_pose = tf.identity(scaled_logits_pose, name=is_training_prefix+'scaled_logits')
  rot_q_error_map = tf.identity(rot_q_error_map, name=is_training_prefix+'rot_error_map')
  trans_error_map = tf.identity(trans_error_map, name=is_training_prefix+'trans_error_map')
  ## Regression loss for shape
  balance_shape_loss = 1.
  _, scaled_logits_shape = train_utils.add_l1_regression_loss(
          tf.gather(reg_logits_concat, range(7, dataset.SHAPE_DIMS+7), axis=3),
          samples['shape_map'],
          samples['mask'],
          balance=balance_shape_loss,
          upsample_logits=FLAGS.upsample_logits,
          name=is_training_prefix + 'loss_reg_shape',
          loss_collection=tf.GraphKeys.LOSSES if is_training else None
          )
  scaled_logits = tf.concat([scaled_logits_pose, scaled_logits_shape], axis=3)

  bin_range = [np.linspace(r[0], r[1], num=b).tolist() for r, b in zip(dataset.pose_range, dataset.bin_nums)]
  label_id_list = []
  loss_slice_crossentropy_list = []
  for idx_output, output in enumerate(dataset.output_names):
    # Get label_id slice
    label_slice = tf.gather(samples['pose_shape_map'], [idx_output], axis=3)
    bin_vals_output = bin_range[idx_output]
    label_id_slice = tf.round((label_slice - bin_vals_output[0]) / (bin_vals_output[1] - bin_vals_output[0]))
    label_id_slice = tf.clip_by_value(label_id_slice, 0, dataset.bin_nums[idx_output]-1)
    label_id_slice = tf.cast(label_id_slice, tf.uint8)
    label_id_list.append(label_id_slice)

    # Add losses for each output names for logging
    scaled_logits_slice = tf.gather(scaled_logits, [idx_output], axis=3)
    scaled_logits_slice_masked = tf.where(masks, scaled_logits_slice, tf.zeros_like(scaled_logits_slice))
    loss_slice_reg = tf.losses.huber_loss(label_slice, scaled_logits_slice_masked, delta=1.0, loss_collection=None) / count_valid
    loss_slice_reg = tf.identity(loss_slice_reg, name=is_training_prefix+'loss_slice_reg_'+output)

    ## Cross-entropy loss for each output http://icode.baidu.com/repos/baidu/personal-code/video_seg_transfer/blob/with_db:Networks/mx_losses.py (L89)
    balance_cls_loss = 1e-1
    scaled_logits_disc_slice, _ = train_utils.scale_logits_to_labels(outputs_to_logits[output], label_slice, True)
    neg_log = -1. * tf.nn.log_softmax(scaled_logits_disc_slice)
    gt_idx = tf.one_hot(tf.squeeze(label_id_slice), depth=dataset.bin_nums[idx_output], axis=-1)
    loss_slice_crossentropy = tf.reduce_sum(tf.multiply(gt_idx, neg_log), axis=3, keepdims=True)
    loss_slice_crossentropy = tf.where(masks, loss_slice_crossentropy, tf.zeros_like(loss_slice_crossentropy))
    loss_slice_crossentropy= tf.reduce_sum(loss_slice_crossentropy) / count_valid * balance_cls_loss
    loss_slice_crossentropy = tf.identity(loss_slice_crossentropy, name=is_training_prefix+'loss_slice_cls_'+output)
    loss_slice_crossentropy_list.append(loss_slice_crossentropy)
    if is_training and idx_output>=7:
        tf.losses.add_loss(loss_slice_crossentropy, loss_collection=tf.GraphKeys.LOSSES)
  loss_crossentropy = tf.identity(tf.add_n(loss_slice_crossentropy_list), name=is_training_prefix+'loss_cls_ALL')
  label_id = tf.concat(label_id_list, axis=3)
  label_id_masked = tf.where(tf.tile(masks, [1, 1, 1, tf.shape(label_id)[3]]), label_id, tf.zeros_like(label_id))
  label_id_masked = tf.identity(label_id_masked, name=is_training_prefix+'label_id')

  ## Regression loss for shape
  # balance_shape_loss = 1e-3
  # shape_logits = outputs_to_logits['shape']
  # shape_loss, scaled_shape_logits = train_utils.add_l1_regression_loss(
  #         shape_logits,
  #         samples['shape_map'],
  #         samples['mask'],
  #         balance=balance_shape_loss,
  #         upsample_logits=FLAGS.upsample_logits,
  #         loss_collection=None
  #         )
  # scaled_shape_logits = tf.identity(scaled_shape_logits, name=is_training_prefix+'scaled_shape_logits')
  # shape_loss = tf.identity(shape_loss, name=is_training_prefix + 'loss_all_shape')
  # # tf.losses.add_loss(shape_loss, loss_collection=tf.GraphKeys.LOSSES)

  ## Classification to 79 loss for shape
  # shape_id_map_logits = outputs_to_logits['shape_id_map']
  # scaled_shape_id_map_logits, _ = train_utils.scale_logits_to_labels(shape_id_map_logits, samples['shape_id_map'], True)
  # gt_idx = tf.one_hot(tf.squeeze(samples['shape_id_map_gt']), depth=79, axis=-1)
  # loss_shape_crossentropy = tf.losses.softmax_cross_entropy(gt_idx, scaled_shape_id_map_logits, weights=tf.to_float(tf.squeeze(masks)), loss_collection=None)
  # loss_shape_crossentropy = tf.identity(loss_shape_crossentropy, name=is_training_prefix+'loss_all_shape_id_cls')
  # tf.losses.add_loss(loss_shape_crossentropy, loss_collection=tf.GraphKeys.LOSSES)

  # shape_id_map_predicts = tf.expand_dims(tf.argmax(scaled_shape_id_map_logits, axis=3), axis=-1)

  shape_sim_mat = np.loadtxt('./deeplab/dataset-api/car_instance/sim_mat.txt')
  assert shape_sim_mat.shape[0] == shape_sim_mat.shape[1]
  num_cads = shape_sim_mat.shape[0]
  scaled_logits_shape_expanded = tf.tile(tf.expand_dims(scaled_logits_shape, axis=3), [1, 1, 1, num_cads, 1])
  codes = np.load('/home/share/zhurui/Documents/tf-models-official/research/deeplab/datasets/apolloscape/codes.npy')
  codes_cons = tf.constant(np.transpose(codes), dtype=tf.float32) # [79, 10]
  codes_expanded = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(codes_cons, 0), 0), 0), [tf.shape(scaled_logits_shape)[0], tf.shape(scaled_logits_shape)[1], tf.shape(scaled_logits_shape)[2], 1, 1])
  shape_l2_error_per_cls = tf.reduce_sum(tf.square(scaled_logits_shape_expanded - codes_expanded), axis=4)
  shape_id_map_predicts = tf.expand_dims(tf.argmin(shape_l2_error_per_cls, axis=3), axis=-1)

  print 'oooo', shape_l2_error_per_cls.get_shape(), shape_id_map_predicts.get_shape()
  # argmax_logits_list = []
  # for idx in range(dataset.SHAPE_DIMS):
  #   scaled_logits_shape_slice = tf.gather(scaled_logits_shape, [idx], axis=3)
  #   argmax_logits = train_utils.logits_cls_to_logits_id(scaled_logits_shape_slice)
  #   argmax_logits_list.append(argmax_logits)
  # shape_id_map_predicts = tf.concat(argmax_logits_list, axis=3)
  shape_id_map_predicts = tf.identity(shape_id_map_predicts, name=is_training_prefix + 'shape_id_map_predict')

  shape_id_map_labels_flattened = tf.boolean_mask(samples['shape_id_map_gt'], masks)
  shape_id_map_predicts_flattened = tf.boolean_mask(shape_id_map_predicts, masks)
  shape_id_map_errors = tf.gather_nd(tf.constant(shape_sim_mat, dtype=tf.float32),
          tf.stack([shape_id_map_labels_flattened, shape_id_map_predicts_flattened], axis=1))
  shape_cls_metric_loss = tf.identity(tf.reduce_mean(shape_id_map_errors), name=is_training_prefix + 'loss_all_shape_id_cls_metric')

  shape_cls_metric_error_map = tf.gather_nd(tf.constant(shape_sim_mat, dtype=tf.float32),
          tf.stack([samples['shape_id_map_gt'], shape_id_map_predicts], axis=-1))
  shape_cls_metric_error_map = tf.where(masks, shape_cls_metric_error_map, tf.zeros_like(shape_cls_metric_error_map))
  shape_cls_metric_error_map = tf.identity(shape_cls_metric_error_map, name=is_training_prefix + 'shape_id_cls_error_map')

  shape_cls_metric_loss_check = tf.reduce_sum(shape_cls_metric_error_map) / count_valid
  shape_cls_metric_loss_check = tf.identity(shape_cls_metric_loss_check, name=is_training_prefix + 'loss_all_shape_id_cls_metric_check')



