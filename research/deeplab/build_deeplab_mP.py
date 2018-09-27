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

  outputs_to_logits, outputs_to_logits_map, outputs_to_weights_map, outputs_to_areas_N = model.single_scale_logits(
    FLAGS,
    samples[common.IMAGE],
    samples['seg_rescaled'],
    samples['car_nums'],
    idx_xys,
    bin_vals,
    outputs_to_indices,
    model_options=model_options,
    weight_decay=FLAGS.weight_decay,
    is_training=is_training,
    fine_tune_batch_norm=FLAGS.fine_tune_batch_norm and is_training,
    fine_tune_feature_extractor=FLAGS.fine_tune_feature_extractor and is_training)
  print outputs_to_logits

  # Get regressed logits for all outputs
  reg_logits_list = []

  for output in dataset.output_names:
      if output not in ['x', 'y'] or not(FLAGS.if_uvflow):
          prob_logits = train_utils.logits_cls_to_logits_probReg(
              outputs_to_logits[output],
              bin_vals[outputs_to_indices[output]]) # [car_num_total, 1]
          reg_logits_list.append(prob_logits)
          print '||||||||CLS logits for '+output
      else:
          reg_logits_list.append(outputs_to_logits[output])
          print '||||||||REG logits for '+output
      outputs_to_weights_map[output] = tf.identity(outputs_to_weights_map[output], name=is_training_prefix+'%s_weights_map'%output)
  reg_logits_concat = tf.concat(reg_logits_list, axis=1) # [car_num_total, 17]
  reg_logits_concat = tf.where(tf.is_nan(reg_logits_concat), tf.zeros_like(reg_logits_concat)+1e-5, reg_logits_concat) # Hack to process NaN!!!!!!
  reg_logits_mask = tf.logical_not(tf.is_nan(tf.reduce_sum(reg_logits_concat, axis=1, keepdims=True)))
  # for output in dataset.output_names: # should be identical for all outputt
  areas_masked = outputs_to_areas_N[dataset.output_names[0]]
  # pixels_valid = tf.reduce_sum(areas_masked)+1e-10
  masks_float = tf.to_float(tf.not_equal(areas_masked, 0.))
  # weights_normalized = areas_masked # weights equals area; will be divided by num of all pixels later
  weights_normalized = tf.ones_like(areas_masked) # NOT weights equals area

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
  balance_trans_reg_loss = 10.
  pose_dict_N = tf.gather_nd(samples['pose_dict'], idx_xys) # [N, 7]
  pose_dict_N = tf.identity(pose_dict_N, is_training_prefix+'pose_dict_N')
  rotuvd_dict_N = tf.gather_nd(samples['rotuvd_dict'], idx_xys) # [N, 7]
  rotuvd_dict_N = tf.identity(rotuvd_dict_N, is_training_prefix+'rotuvd_dict_N')
  def within_frame(W, H, u, v):
      within_mask = tf.concat([tf.greater_equal(u, 0), tf.greater_equal(v, 0),
          tf.less_equal(u, W), tf.less_equal(v, H)], axis=1)
      return tf.reduce_all(within_mask, axis=1, keepdims=True)
  def within_range(x, min_clip, max_clip):
      within_mask = tf.logical_and(tf.greater_equal(x, min_clip), tf.less_equal(x, max_clip))
      return within_mask
  rotuvd_dict_N_within = tf.to_float(within_frame(680, 544, tf.gather(rotuvd_dict_N, [4], axis=1), tf.gather(rotuvd_dict_N, [5], axis=1)))
  # bbox_dict_N = tf.gather_nd(samples['bbox_dict'], idx_xys) # [N, 4], [x_min, x_max, y_min, y_max]
  # bbox_dict_N = tf.identity(bbox_dict_N, is_training_prefix+'bbox_dict_N')
  # # bbox_c_dict_N2 = tf.concat([0.5*(tf.gather(bbox_dict_N, [0], axis=1)+tf.gather(bbox_dict_N, [1], axis=1)),
  #     # 0.5*(tf.gather(bbox_dict_N, [2], axis=1)+tf.gather(bbox_dict_N, [3], axis=1))], axis=1)
  # bbox_c_dict_N = tf.gather_nd(samples['bbox_c_dict'], idx_xys) # [N, 2]
  # bbox_c_dict_N = tf.identity(bbox_c_dict_N, is_training_prefix+'bbox_c_dict_N')
  # bbox_c_dict_N_within = tf.to_float(within_frame(680, 544,
  #     tf.gather(bbox_c_dict_N, [0], axis=1), tf.gather(bbox_c_dict_N, [1], axis=1)))
  # quat_deltauv_dinvd_dict_N = tf.gather_nd(samples['quat_deltauv_dinvd_dict'], idx_xys) # [N, 7]; NEW GT with offsets
  # quat_deltauv_dinvd_dict_N = tf.identity(quat_deltauv_dinvd_dict_N, is_training_prefix+'quat_deltauv_dinvd_dict_N')
  # quat_deltauv_dinvd_dict_N_within = tf.to_float(tf.logical_and(
  #     within_range(tf.gather(quat_deltauv_dinvd_dict_N, [4], axis=1), -30., 30.),
  #     within_range(tf.gather(quat_deltauv_dinvd_dict_N, [5], axis=1), -20., 5.)
  #     ))
  # masks_float = masks_float * rotuvd_dict_N_within * bbox_c_dict_N_within * quat_deltauv_dinvd_dict_N_within
  masks_float = masks_float * rotuvd_dict_N_within # [N, 1]
  weights_normalized = masks_float * weights_normalized
  count_valid = tf.reduce_sum(masks_float)+1e-10
  pixels_valid = tf.reduce_sum(areas_masked * masks_float)+1e-10

  def rotuvd_dict_N_2_quat_xy_dinvd_dict_N(rotuvd_dict_N_input):
      u = tf.gather(rotuvd_dict_N_input, [4], axis=1)
      v = tf.gather(rotuvd_dict_N_input, [5], axis=1)
      d = tf.gather(rotuvd_dict_N_input, [6], axis=1)
      F1 = 463.08880615234375
      W = 338.8421325683594
      F2 = 462.87689208984375
      H = 271.9969482421875
      K_T = tf.constant([[1./F1, 0., -W/F1], [0, 1./F2, -H/F2], [0., 0., 1.]])
      if FLAGS.if_depth:
        uvd = tf.concat([u*d, v*d, d], axis=1)
      else:
        uvd = tf.concat([u/d, v/d, 1/d], axis=1)
      xyz = tf.transpose(tf.matmul(K_T, tf.transpose(uvd))) # x, y, depth; [N, 3]
      quat_xy_dinvd_dict_N_output = tf.concat([tf.gather(rotuvd_dict_N_input, [0,1,2,3], axis=1), tf.gather(xyz, [0,1], axis=1), d], axis=1)
      return quat_xy_dinvd_dict_N_output

  # def quat_deltauv_dinvd_dict_N_2_quat_xy_dinvd_dict_N(quat_deltauv_dinvd_dict_N_input):
  #     u = tf.gather(quat_deltauv_dinvd_dict_N_input, [4], axis=1) + tf.gather(bbox_c_dict_N, [0], axis=1)
  #     v = tf.gather(quat_deltauv_dinvd_dict_N_input, [5], axis=1) + tf.gather(bbox_c_dict_N, [1], axis=1)
  #     d = tf.gather(quat_deltauv_dinvd_dict_N_input, [6], axis=1)
  #     F1 = 463.08880615234375
  #     W = 338.8421325683594
  #     F2 = 462.87689208984375
  #     H = 271.9969482421875
  #     K_T = tf.constant([[1./F1, 0., -W/F1], [0, 1./F2, -H/F2], [0., 0., 1.]])
  #     if FLAGS.if_depth:
  #       uvd = tf.concat([u*d, v*d, d], axis=1)
  #     else:
  #       uvd = tf.concat([u/d, v/d, 1/d], axis=1)
  #     xyz = tf.transpose(tf.matmul(K_T, tf.transpose(uvd))) # x, y, depth; [N, 3]
  #     quat_xy_dinvd_dict_N = tf.concat([tf.gather(quat_deltauv_dinvd_dict_N_input, [0,1,2,3], axis=1), tf.gather(xyz, [0,1], axis=1), d], axis=1)
  #     return quat_xy_dinvd_dict_N


  prob_logits_pose = tf.gather(reg_logits_concat, [0, 1, 2, 3, 4, 5, 6], axis=1)
  prob_logits_pose = tf.identity(prob_logits_pose, name=is_training_prefix+'prob_logits_pose')
  if FLAGS.if_uvflow:
      prob_logits_pose_xy_from_uv = rotuvd_dict_N_2_quat_xy_dinvd_dict_N(prob_logits_pose)
  # return pose_dict_N, rotuvd_dict_N_2_quat_xy_dinvd_dict_N(rotuvd_dict_N)


  _, prob_logits_pose, rot_q_error_cars, trans_l2, depth_diff_abs, depth_relative = train_utils.add_my_pose_loss_cars(
          prob_logits_pose,
          rotuvd_dict_N if FLAGS.if_uvflow else pose_dict_N,
          prob_logits_pose_xy_from_uv if FLAGS.if_uvflow else prob_logits_pose,
          pose_dict_N,
          masks_float,
          weights_normalized,
          balance_rot=balance_rot_reg_loss,
          balance_trans=balance_trans_reg_loss,
          upsample_logits=FLAGS.upsample_logits,
          name=is_training_prefix + 'loss_reg',
          is_training_prefix = is_training_prefix,
          loss_collection=tf.GraphKeys.LOSSES if is_training else None,
          if_depth=FLAGS.if_depth)
  # rot_q_error_map = tf.identity(logits_cars_to_map(rot_q_error_cars), name=is_training_prefix+'rot_error_map')
  # trans_error_map = tf.identity(logits_cars_to_map(trans_error_cars), name=is_training_prefix+'trans_error_map')
  trans_l2 = tf.identity(trans_l2, name=is_training_prefix+'trans_l2')
  depth_diff_abs = tf.identity(depth_diff_abs, name=is_training_prefix+'depth_diff_abs')
  depth_relative = tf.identity(depth_relative, name=is_training_prefix+'depth_relative')

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
          loss_collection=tf.GraphKeys.LOSSES if (is_training and FLAGS.if_shape) else None
          )
  prob_logits_pose_shape = tf.concat([prob_logits_pose, prob_logits_shape], axis=1)
  prob_logits_pose_shape = tf.identity(prob_logits_pose_shape, name=is_training_prefix+'prob_logits_pose_shape_cars')
  if FLAGS.if_uvflow:
      pose_shape_dict_N = tf.concat([rotuvd_dict_N, shape_dict_N], axis=1)
  else:
      pose_shape_dict_N = tf.concat([pose_dict_N, shape_dict_N], axis=1)

  masks_map_filtered = tf.identity(logits_cars_to_map(masks_float), name=is_training_prefix+'masks_map_filtered')
  if FLAGS.save_summaries_images:
    prob_logits_pose_shape_map = logits_cars_to_map(prob_logits_pose_shape)
    prob_logits_pose_shape_map = tf.identity(prob_logits_pose_shape_map, name=is_training_prefix+'prob_logits_pose_shape_map')
    label_pose_shape_map = logits_cars_to_map(pose_shape_dict_N)
    label_pose_shape_map = tf.identity(label_pose_shape_map, name=is_training_prefix+'label_pose_shape_map')

    if FLAGS.if_uvflow:
        label_uv_map = tf.gather(label_pose_shape_map, [4, 5], axis=3) # (-1, 272, 680, 2)
        v_coords = tf.range(tf.shape(label_uv_map)[1])
        u_coords = tf.range(tf.shape(label_uv_map)[2])
        Vs, Us = tf.meshgrid(v_coords, u_coords)
        features_Ys = tf.tile(tf.expand_dims(tf.expand_dims(tf.transpose(Vs + tf.shape(label_uv_map)[1]), -1), 0), [tf.shape(label_uv_map)[0], 1, 1, 1]) # Adding half height because of the crop!
        features_Xs = tf.tile(tf.expand_dims(tf.expand_dims(tf.transpose(Us), -1), 0), [tf.shape(label_uv_map)[0], 1, 1, 1])
        coords_UVs_concat = tf.to_float(tf.concat([features_Xs, features_Ys], axis=3))
        label_uv_flow_map = tf.multiply(masks_map_filtered, label_uv_map - coords_UVs_concat)
        label_uv_flow_map = tf.identity(label_uv_flow_map, name=is_training_prefix+'label_uv_flow_map')

        logits_uv_flow_map = tf.concat([outputs_to_logits_map['x'], outputs_to_logits_map['y']], axis=3)
        masks_map_filtered_rescaled = tf.image.resize_nearest_neighbor(masks_map_filtered, [tf.shape(logits_uv_flow_map)[1], tf.shape(logits_uv_flow_map)[2]], align_corners=True)
        logits_uv_flow_map = tf.identity(tf.multiply(masks_map_filtered_rescaled, logits_uv_flow_map), name=is_training_prefix+'logits_uv_flow_map')

        # label_uv_map_rescaled = logits_cars_to_map(tf.gather(rotuvd_dict_N, [4, 5], axis=3) # (-1, 68, 170, 2)
        # v_coords_rescaled = tf.range(tf.shape(label_uv_map_rescaled)[1])
        # u_coords_rescaled = tf.range(tf.shape(label_uv_map_rescaled)[2])
        # Vs_rescaled, Us_rescaled = tf.meshgrid(v_coords_rescaled, u_coords_rescaled)
        # features_Ys_rescaled = tf.tile(tf.expand_dims(tf.expand_dims(tf.transpose(Vs_rescaled + tf.shape(label_uv_map_rescaled)[1]), -1), 0), [tf.shape(label_uv_map_rescaled)[0], 1, 1, 1]) # Adding half height because of the crop!
        # features_Xs_rescaled = tf.tile(tf.expand_dims(tf.expand_dims(tf.transpose(Us_rescaled), -1), 0), [tf.shape(label_uv_map_rescaled)[0], 1, 1, 1])
        # coords_UVs_concat_rescaled = tf.to_float(tf.concat([features_Xs_rescaled * model_options.decoder_output_stride,
        #     features_Ys_rescaled * model_options.decoder_output_stride], axis=3))
        # label_uv_flow_map_rescaled = label_uv_map_rescaled - coords_UVs_concat_rescaled
        # label_uv_flow_map_rescaled = tf.identity(label_uv_flow_map_rescaled, name=is_training_prefix+'label_uv_flow_map_rescaled')

        balance_uv_flow = 1.
        label_uv_flow_map_rescaled = tf.image.resize_nearest_neighbor(label_uv_flow_map, [tf.shape(logits_uv_flow_map)[1], tf.shape(logits_uv_flow_map)[2]], align_corners=True)
        pixels_valid_filtered = tf.reduce_sum(masks_map_filtered_rescaled)+1e-10
        uv_flow_map_diff = tf.multiply(tf.abs(logits_uv_flow_map - label_uv_flow_map_rescaled), masks_map_filtered_rescaled)
        loss_reg_uv_flow_map = tf.reduce_sum(uv_flow_map_diff) / pixels_valid_filtered * balance_uv_flow # L1 loss for uv flow
        loss_reg_uv_flow_map = tf.identity(loss_reg_uv_flow_map, name=is_training_prefix+'loss_reg_uv_flow_map')
        if is_training:
            tf.losses.add_loss(loss_reg_uv_flow_map, loss_collection=tf.GraphKeys.LOSSES)

  label_id_list = []
  loss_slice_crossentropy_list = []
  for idx_output, output in enumerate(dataset.output_names):
    if idx_output not in [0, 1, 2, 3, 4,5,6] and not(FLAGS.if_shape): # not adding SHAPE loss
        continue

    label_slice = tf.gather(pose_shape_dict_N, [idx_output], axis=1)

    # Add losses for each output names for logging
    prob_logits_slice = tf.gather(prob_logits_pose_shape, [idx_output], axis=1)
    # print 'Logging reg error for: ', idx_output, output
    loss_slice_reg = tf.reduce_sum(tf.abs(label_slice - prob_logits_slice)) / count_valid # [N, 1]; L1 error
    loss_slice_reg = tf.identity(loss_slice_reg, name=is_training_prefix+'loss_slice_reg_'+output)

    ## Cross-entropy loss for each output http://icode.baidu.com/repos/baidu/personal-code/video_seg_transfer/blob/with_db:Networks/mx_losses.py (L89)
    balance_cls_loss = 1.
    if output not in ['x', 'y'] or not(FLAGS.if_uvflow):
        print '... adding cls loss for: ', idx_output, output
        bin_vals_output = bin_range[idx_output]
        label_id_slice = tf.round((label_slice - bin_vals_output[0]) / (bin_vals_output[1] - bin_vals_output[0]))
        label_id_slice = tf.clip_by_value(label_id_slice, 0, dataset.bin_nums[idx_output]-1)
        label_id_slice = tf.cast(label_id_slice, tf.uint8)
        # label_id_list.append(label_id_slice)
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
  # label_id = tf.concat(label_id_list, axis=1)
  # label_id_map = logits_cars_to_map(label_id)
  # label_id_map = tf.identity(label_id_map, name=is_training_prefix+'pose_shape_label_id_map')

  if FLAGS.if_summary_shape_metrics and FLAGS.if_shape:
      shape_sim_mat = np.loadtxt('./deeplab/dataset-api/car_instance/sim_mat.txt')
      assert shape_sim_mat.shape[0] == shape_sim_mat.shape[1]
      num_cads = shape_sim_mat.shape[0]
      prob_logits_shape_expanded = tf.tile(tf.expand_dims(prob_logits_shape, axis=1), [1, num_cads, 1])
      codes_cons = tf.constant(np.transpose(codes), dtype=tf.float32) # [79, 10]
      codes_expanded = tf.tile(tf.expand_dims(codes_cons, 0), [tf.shape(prob_logits_shape_expanded)[0], 1, 1])
      shape_l2_error_per_cls = tf.reduce_sum(tf.square(prob_logits_shape_expanded - codes_expanded), axis=2)
      shape_id_map_predicts = tf.expand_dims(tf.argmin(shape_l2_error_per_cls, axis=1), axis=-1) # [num_cars, 1]

      shape_id_dict_N = tf.gather_nd(samples['shape_id_dict'], idx_xys)
      shape_cls_metric_error_cars = tf.gather_nd(tf.constant(shape_sim_mat, dtype=tf.float32),
              tf.stack([shape_id_dict_N, shape_id_map_predicts], axis=-1)) # [num_cars, 1]
      if FLAGS.save_summaries_images:
        shape_cls_metric_error_map = tf.identity(logits_cars_to_map(shape_cls_metric_error_cars), name=is_training_prefix+'shape_id_sim_map')

      shape_cls_metric_loss_check = tf.reduce_sum(shape_cls_metric_error_cars * masks_float) / count_valid
      shape_cls_metric_loss_check = tf.identity(shape_cls_metric_loss_check, name=is_training_prefix+'loss_all_shape_id_cls_metric')

  return samples[common.IMAGE_NAME], outputs_to_logits['z'], outputs_to_weights_map, seg_one_hots_list, weights_normalized, samples['car_nums'], car_nums_list, idx_xys, prob_logits_pose_xy_from_uv if FLAGS.if_uvflow else prob_logits_pose, pose_dict_N, prob_logits_pose, rotuvd_dict_N, masks_float, tf.multiply(masks_map_filtered, label_uv_flow_map) if FLAGS.if_uvflow else masks_map_filtered, tf.multiply(masks_map_filtered_rescaled, logits_uv_flow_map) if FLAGS.if_uvflow else masks_map_filtered
  # , shape_cls_metric_error_cars, count_valid
