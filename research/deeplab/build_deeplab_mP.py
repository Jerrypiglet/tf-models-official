import tensorflow as tf
from deeplab import common
from deeplab import model_maskLogits as model
from deeplab.utils import train_utils_mP as train_utils
import numpy as np

def _build_deeplab(FLAGS, samples, outputs_to_num_classes, outputs_to_indices, bin_vals, bin_range, dataset, codes, is_training=True):
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
  if is_training:
      is_training_prefix = ''
  else:
      is_training_prefix = 'val-'

  # Add name to input and label nodes so we can add to summary.
  model_options = common.ModelOptions(
    outputs_to_num_classes=outputs_to_num_classes,
    crop_size=[dataset.height, dataset.width],
    atrous_rates=FLAGS.atrous_rates,
    output_stride=FLAGS.output_stride)

  samples[common.IMAGE] = tf.identity(
      samples[common.IMAGE], name=is_training_prefix+common.IMAGE)
  samples[common.IMAGE_NAME] = tf.identity(
      samples[common.IMAGE_NAME], name=is_training_prefix+common.IMAGE_NAME)
  samples['seg'] = tf.identity(samples['seg'], name=is_training_prefix+'seg')
  masks = tf.identity(samples['mask'], name=is_training_prefix+'not_ignore_mask_in_loss')
  masks_rescaled_float = tf.identity(samples['mask_rescaled_float'], name=is_training_prefix+'mask_rescaled_float')

  if FLAGS.val_split != 'test':
    samples['vis'] = tf.identity(samples['vis'], name=is_training_prefix+'vis')
    # samples['pose_map'] = tf.identity(samples['pose_map'], name=is_training_prefix+'pose_map')
    # samples['shape_map'] = tf.identity(samples['shape_map'], name=is_training_prefix+'shape_map')
    samples['pose_dict'] = tf.identity(samples['pose_dict'], name=is_training_prefix+'pose_dict')
    samples['rotuvd_dict'] = tf.identity(samples['rotuvd_dict'], name=is_training_prefix+'rotuvd_dict')
    samples['bbox_dict'] = tf.identity(samples['bbox_dict'], name=is_training_prefix+'bbox_dict')
    samples['shape_dict'] = tf.identity(samples['shape_dict'], name=is_training_prefix+'shape_dict')

    car_nums_list = tf.split(samples['car_nums'], samples['car_nums'].get_shape()[0], axis=0)

    def _unpadding(padded, append_left_idx=False): # input: [batch_size, ?, D], output: [car_num_total, D]
        padded_list = tf.split(padded, padded.get_shape()[0])
        unpadded_list = []
        for idx, (padded, car_num) in enumerate(zip(padded_list, car_nums_list)):
          unpadded = tf.slice(tf.squeeze(padded, 0), [0, 0], [tf.squeeze(car_num), padded.get_shape()[2]])
          if append_left_idx:
            unpadded = tf.concat([tf.zeros_like(unpadded)+idx, unpadded], axis=1)
          unpadded_list.append(unpadded)
        unpadded = tf.concat(unpadded_list, axis=0) # (car_num_total=?_1+...+?_bs, D)
        return unpadded

    idx_xys = _unpadding(samples['idxs'], True) # [N, 2]

    # seg_one_hots_N_flattened = tf.tf.gather_nd(samples['seg_one_hots_flattened'], idx_xys) # [N, 272/4*680/4], tf.int32

    seg_one_hots_list = []
    seg_one_hots_flattened_list = []
    seg_list = tf.split(samples['seg'], samples['seg'].get_shape()[0])
    for seg_sample, car_num in zip(seg_list, car_nums_list):
      seg_one_hots_sample = tf.one_hot(tf.squeeze(tf.cast(seg_sample, tf.int32)), depth=tf.reshape(car_num+1, []))
      seg_one_hots_sample = tf.slice(seg_one_hots_sample, [0, 0, 1], [-1, -1, -1])
      seg_one_hots_list.append(tf.expand_dims(seg_one_hots_sample, 0)) # (1, 272, 680, ?)
      seg_one_hots_flattened_list.append(tf.reshape(seg_one_hots_sample, [-1, tf.shape(seg_one_hots_sample)[2]])) # (272*680, ?)

  def logits_cars_to_map(logits_cars):
    logits_cars_N_list = tf.split(logits_cars, samples['car_nums'])
    logits_samples_list = []
    for seg_one_hots_sample, logits_cars_sample in zip(seg_one_hots_flattened_list, logits_cars_N_list): # (272*680, ?) (?, 17)
      logits_sample = tf.matmul(seg_one_hots_sample, tf.cast(logits_cars_sample, seg_one_hots_sample.dtype))
      logits_sample  = tf.cast(logits_sample, logits_cars_sample.dtype)
      logits_samples_list.append(tf.reshape(logits_sample, [dataset.height, dataset.width, logits_cars_sample.get_shape()[1]]))
    logits_map = tf.stack(logits_samples_list, axis=0) # (3, 272, 680, 17)
    return logits_map

  outputs_to_logits, outputs_to_weights_map, outputs_to_areas_N = model.single_scale_logits(
    samples[common.IMAGE],
    samples['seg_rescaled'],
    samples['car_nums'],
    idx_xys,
    model_options=model_options,
    weight_decay=FLAGS.weight_decay,
    is_training=is_training,
    fine_tune_batch_norm=FLAGS.fine_tune_batch_norm and is_training,
    fine_tune_feature_extractor=FLAGS.fine_tune_feature_extractor and is_training)
  print outputs_to_logits

  # Get regressed logits for all outputs
  reg_logits_list = []

  for output in dataset.output_names:
      prob_logits = train_utils.logits_cls_to_logits_probReg(
          outputs_to_logits[output],
          bin_vals[outputs_to_indices[output]]) # [car_num_total, 1]
      reg_logits_list.append(prob_logits)
  reg_logits_concat = tf.concat(reg_logits_list, axis=1) # [car_num_total, 17]
  reg_logits_concat = tf.where(tf.is_nan(reg_logits_concat), tf.zeros_like(reg_logits_concat)+1e-5, reg_logits_concat) # Hack to process NaN!!!!!!
  reg_logits_mask = tf.logical_not(tf.is_nan(tf.reduce_sum(reg_logits_concat, axis=1, keepdims=True)))
  # for output in dataset.output_names: # should be identical for all outputt
  areas_masked = outputs_to_areas_N[dataset.output_names[0]]
  pixels_valid = tf.reduce_sum(areas_masked)+1e-10
  masks_float = tf.to_float(tf.not_equal(areas_masked, 0.))
  count_valid = tf.reduce_sum(masks_float)+1e-10
  # weights_normalized = tf.to_float(tf.shape(weights_masked)[0]) * weights_masked / tf.reduce_sum(weights_masked) # [N, 1]. sum should be N in for N cars in the batch
  weights_normalized = areas_masked # weights equals area; will be divided by num of all pixels later

  # if FLAGS.val_split == 'test':
  #     scaled_prob_logits_pose = train_utils.scale_for_l1_loss(
  #             tf.gather(reg_logits_concat, [0, 1, 2, 3, 4, 5, 6], axis=3), samples['mask'], samples['mask'], upsample_logits=FLAGS.upsample_logits)
  #     scaled_prob_logits_shape = train_utils.scale_for_l1_loss(
  #             tf.gather(reg_logits_concat, range(7, dataset.SHAPE_DIMS+7), axis=3), samples['mask'], samples['mask'], upsample_logits=FLAGS.upsample_logits)
  #     scaled_prob_logits = tf.concat([scaled_prob_logits_pose, scaled_prob_logits_shape], axis=3)
  #     scaled_prob_logits = tf.identity(scaled_prob_logits, name=is_training_prefix+'scaled_prob_logits_pose_shape_map')
  # return samples['idxs']

  ## Regression loss for pose
  balance_rot_reg_loss = 1.
  balance_trans_reg_loss = 1.
  pose_dict_N = tf.gather_nd(samples['pose_dict'], idx_xys) # [N, 7]
  pose_dict_N = tf.identity(pose_dict_N, is_training_prefix+'pose_dict_N')
  rotuvd_dict_N = tf.gather_nd(samples['rotuvd_dict'], idx_xys) # [N, 6]
  rotuvd_dict_N = tf.identity(rotuvd_dict_N, is_training_prefix+'rotuvd_dict_N')

  return pose_dict_N, rotuvd_dict_N

  _, prob_logits_pose, rot_q_error_cars, trans_error_cars = train_utils.add_my_pose_loss_cars(
          tf.gather(reg_logits_concat, [0, 1, 2, 3, 4, 5, 6], axis=1),
          pose_dict_N,
          masks_float,
          weights_normalized,
          balance_rot=balance_rot_reg_loss,
          balance_trans=balance_trans_reg_loss,
          upsample_logits=FLAGS.upsample_logits,
          name=is_training_prefix + 'loss_reg',
          loss_collection=tf.GraphKeys.LOSSES if is_training else None)
  prob_logits_pose = tf.identity(tf.gather(reg_logits_concat, [0, 1, 2, 3, 4, 5, 6], axis=1), name=is_training_prefix+'prob_logits_pose')
  rot_q_error_map = tf.identity(logits_cars_to_map(rot_q_error_cars), name=is_training_prefix+'rot_error_map')
  trans_error_map = tf.identity(logits_cars_to_map(trans_error_cars), name=is_training_prefix+'trans_error_map')

  ## Regression loss for shape
  balance_shape_loss = 1.
  shape_dict_N = tf.gather_nd(samples['shape_dict'], idx_xys)
  _, prob_logits_shape = train_utils.add_l1_regression_loss_cars(
          tf.gather(reg_logits_concat, range(7, dataset.SHAPE_DIMS+7), axis=1),
          shape_dict_N,
          masks_float,
          weights_normalized,
          balance=balance_shape_loss,
          upsample_logits=FLAGS.upsample_logits,
          name=is_training_prefix + 'loss_reg_shape',
          # loss_collection=tf.GraphKeys.LOSSES if is_training else None
          loss_collection=None
          )
  prob_logits_pose_shape = tf.concat([prob_logits_pose, prob_logits_shape], axis=1)
  prob_logits_pose_shape = tf.identity(prob_logits_pose_shape, name=is_training_prefix+'prob_logits_pose_shape_cars')
  pose_shape_dict_N = tf.concat([pose_dict_N, shape_dict_N], axis=1)

  if FLAGS.save_summaries_images:
    prob_logits_pose_shape_map = logits_cars_to_map(prob_logits_pose_shape)
    prob_logits_pose_shape_map = tf.identity(prob_logits_pose_shape_map, name=is_training_prefix+'prob_logits_pose_shape_map')
    # label_pose_shape_map = tf.identity(samples['label_pose_shape_map'], name=is_training_prefix+'label_pose_shape_map')
    label_pose_shape_map = logits_cars_to_map(pose_shape_dict_N)
    label_pose_shape_map = tf.identity(label_pose_shape_map, name=is_training_prefix+'label_pose_shape_map')

  label_id_list = []
  loss_slice_crossentropy_list = []
  for idx_output, output in enumerate(dataset.output_names):
    if idx_output not in [6]:
        continue
    # Get label_id slice
    label_slice = tf.gather(pose_shape_dict_N, [idx_output], axis=1)
    bin_vals_output = bin_range[idx_output]
    label_id_slice = tf.round((label_slice - bin_vals_output[0]) / (bin_vals_output[1] - bin_vals_output[0]))
    label_id_slice = tf.clip_by_value(label_id_slice, 0, dataset.bin_nums[idx_output]-1)
    label_id_slice = tf.cast(label_id_slice, tf.uint8)
    label_id_list.append(label_id_slice)

    # Add losses for each output names for logging
    prob_logits_slice = tf.gather(prob_logits_pose_shape, [idx_output], axis=1)
    print '... adding cls loss and reg lossing for: ', idx_output, output
    loss_slice_reg = tf.reduce_sum(tf.abs(label_slice - prob_logits_slice)) / count_valid # [N, 1]; L1 error
    loss_slice_reg = tf.identity(loss_slice_reg, name=is_training_prefix+'loss_slice_reg_'+output)

    ## Cross-entropy loss for each output http://icode.baidu.com/repos/baidu/personal-code/video_seg_transfer/blob/with_db:Networks/mx_losses.py (L89)
    balance_cls_loss = 1e-1
    neg_log = -1. * tf.nn.log_softmax(outputs_to_logits[output])
    gt_idx = tf.one_hot(tf.squeeze(label_id_slice), depth=dataset.bin_nums[idx_output], axis=-1)
    loss_slice_crossentropy = tf.reduce_sum(tf.multiply(gt_idx, neg_log), axis=1, keepdims=True)
    loss_slice_crossentropy= tf.reduce_sum(tf.multiply(weights_normalized, loss_slice_crossentropy)) / pixels_valid * balance_cls_loss
    loss_slice_crossentropy = tf.identity(loss_slice_crossentropy, name=is_training_prefix+'loss_slice_cls_'+output)
    loss_slice_crossentropy_list.append(loss_slice_crossentropy)
    if is_training:
        # tf.losses.add_loss(loss_slice_crossentropy, loss_collection=None)
        tf.losses.add_loss(loss_slice_crossentropy, loss_collection=tf.GraphKeys.LOSSES)
  loss_crossentropy = tf.identity(tf.add_n(loss_slice_crossentropy_list), name=is_training_prefix+'loss_cls_ALL')
  label_id = tf.concat(label_id_list, axis=1)
  # label_id_map = logits_cars_to_map(label_id)
  # label_id_map = tf.identity(label_id_map, name=is_training_prefix+'pose_shape_label_id_map')

  if FLAGS.if_summary_shape_metrics:
      shape_sim_mat = np.loadtxt('./deeplab/dataset-api/car_instance/sim_mat.txt')
      assert shape_sim_mat.shape[0] == shape_sim_mat.shape[1]
      num_cads = shape_sim_mat.shape[0]
      prob_logits_shape_expanded = tf.tile(tf.expand_dims(prob_logits_shape, axis=1), [1, num_cads, 1])
      codes_cons = tf.constant(np.transpose(codes), dtype=tf.float32) # [79, 10]
      codes_expanded = tf.tile(tf.expand_dims(codes_cons, 0), [tf.shape(prob_logits_shape_expanded)[0], 1, 1])
      shape_l2_error_per_cls = tf.reduce_sum(tf.square(prob_logits_shape_expanded - codes_expanded), axis=2)
      shape_id_map_predicts = tf.expand_dims(tf.argmin(shape_l2_error_per_cls, axis=1), axis=-1) # [num_cars, 1]
      # shape_id_map_predicts = tf.identity(shape_id_map_predicts, name=is_training_prefix + 'shape_id_map_predict')

      shape_id_dict_N = tf.gather_nd(samples['shape_id_dict'], idx_xys)
      shape_cls_metric_error_cars = tf.gather_nd(tf.constant(shape_sim_mat, dtype=tf.float32),
              tf.stack([shape_id_dict_N, shape_id_map_predicts], axis=-1)) # [num_cars, 1]
      if FLAGS.save_summaries_images:
        shape_cls_metric_error_map = tf.identity(logits_cars_to_map(shape_cls_metric_error_cars), name=is_training_prefix+'shape_id_sim_map')

      shape_cls_metric_loss_check = tf.reduce_mean(shape_cls_metric_error_map)
      shape_cls_metric_loss_check = tf.identity(shape_cls_metric_loss_check, name=is_training_prefix+'loss_all_shape_id_cls_metric')

  return samples[common.IMAGE_NAME], outputs_to_logits['z'], outputs_to_weights_map, seg_one_hots_list, weights_normalized, samples['car_nums'], car_nums_list, idx_xys, pose_dict_N, prob_logits_pose, rotuvd_dict_N, masks_float
