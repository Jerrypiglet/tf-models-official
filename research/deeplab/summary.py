import tensorflow as tf
from deeplab import common
from deeplab.utils import train_utils_mP as train_utils
import numpy as np

def get_summaries(FLAGS, graph, summaries, dataset, config, first_clone_scope):
    summary_loss_dict = {}
    if FLAGS.save_summaries_images:
      if FLAGS.num_clones > 1:
          pattern_train = first_clone_scope + '/%s:0'
      else:
          pattern_train = '%s:0'
      pattern_val = 'val-%s:0'
      pattern = pattern_val if FLAGS.if_val else pattern_train
      gather_list_train = range(min(3, int(FLAGS.train_batch_size/FLAGS.num_clones)))
      gather_list_val = range(min(3, int(FLAGS.train_batch_size/FLAGS.num_clones*4)))

      def scale_to_255(tensor, pixel_scaling=None, batch_scale=False):
          tensor = tf.to_float(tensor)
          if pixel_scaling == None:
              if not(batch_scale):
                  offset_to_zero = tf.reduce_min(tf.reduce_min(tf.reduce_min(tensor, axis=3, keepdims=True), axis=2, keepdims=True), axis=1, keepdims=True)
                  scale_to_255 = tf.div(255., tf.reduce_max(tf.reduce_max(tf.reduce_max(
                      tensor - offset_to_zero, axis=3, keepdims=True), axis=2, keepdims=True), axis=1, keepdims=True))
              else:
                  offset_to_zero = tf.reduce_min(tensor)
                  scale_to_255 = tf.div(255., tf.reduce_max(tensor - offset_to_zero))
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
              gather_list = gather_list_train
          else:
              label_postfix = '_val'
              gather_list = gather_list_val
          print gather_list

          summary_mask = graph.get_tensor_by_name(pattern%'not_ignore_mask_in_loss')
          print summary_mask.get_shape()
          summary_mask = tf.reshape(summary_mask, [-1, dataset.height, dataset.width, 1])
          summary_mask_float = tf.to_float(summary_mask)
          summaries.add(tf.summary.image('gt'+label_postfix+'/%s' % 'not_ignore_mask', tf.gather(tf.cast(summary_mask_float*255., tf.uint8), gather_list)))
          print tf.gather(tf.cast(summary_mask_float*255., tf.uint8), gather_list).get_shape()

          summary_mask_filtered = graph.get_tensor_by_name(pattern%'masks_map_filtered')
          summary_mask_filtered = tf.reshape(summary_mask_filtered, [-1, dataset.height, dataset.width, 1])
          summary_mask_float_filtered = tf.to_float(summary_mask_filtered)
          summaries.add(tf.summary.image('gt'+label_postfix+'/%s' % 'masks_map_filtered', tf.gather(tf.cast(summary_mask_float_filtered*255., tf.uint8), gather_list)))

          mask_rescaled_float = graph.get_tensor_by_name(pattern%'mask_rescaled_float')

          seg_outputs = graph.get_tensor_by_name(pattern%'seg')
          summary_seg_output = tf.multiply(summary_mask_float, seg_outputs)
          summary_seg_output_uint8, _ = scale_to_255(summary_seg_output)
          summaries.add(tf.summary.image(
              'gt'+label_postfix+'/seg', tf.gather(summary_seg_output_uint8, gather_list)))

          summary_image = graph.get_tensor_by_name(pattern%common.IMAGE)
          summaries.add(tf.summary.image('gt'+label_postfix+'/%s' % common.IMAGE, tf.gather(summary_image, gather_list)))

          summary_image_name = graph.get_tensor_by_name(pattern%common.IMAGE_NAME)
          summaries.add(tf.summary.text('gt'+label_postfix+'/%s' % common.IMAGE_NAME, tf.gather(summary_image_name, gather_list)))

          summary_vis = graph.get_tensor_by_name(pattern%'vis')
          summaries.add(tf.summary.image('gt'+label_postfix+'/%s' % 'vis', tf.gather(summary_vis, gather_list)))

          if FLAGS.if_depth_only:
              trans_error_names = ['depth_diff_abs_error_map', 'depth_relative_error_map']
          else:
              trans_error_names = ['rot_q_loss_error_map', 'trans_loss_error_map', 'rot_q_angle_error_map', 'trans_sqrt_error_map', 'depth_diff_abs_error_map', 'depth_relative_error_map']
          for error_map_name in trans_error_names:
              summary_error_diffs = graph.get_tensor_by_name(pattern%error_map_name)
              if error_map_name != 'trans_loss_error_map':
                  summary_error_diffs_uint8, _ = scale_to_255(summary_error_diffs, pixel_scaling=None, batch_scale=True)
                  summaries.add(tf.summary.image('metrics_map'+label_postfix+'/%s' % error_map_name, tf.gather(summary_error_diffs_uint8, gather_list)))
                  if error_map_name == 'depth_diff_abs_error_map':
                      summary_error_diffs_less2_8 = tf.where(tf.logical_and(tf.less_equal(summary_error_diffs, 2.8), tf.greater(summary_error_diffs, 0.)), tf.ones_like(summary_error_diffs), tf.zeros_like(summary_error_diffs))
                      summary_error_diffs_uint8, _ = scale_to_255(summary_error_diffs_less2_8, pixel_scaling=None, batch_scale=True)
                      summaries.add(tf.summary.image('metrics_map'+label_postfix+'/depth_diff_abs_error_map_less2.8', tf.gather(summary_error_diffs_uint8, gather_list)))
                      summary_error_diffs_greater2_8 = tf.where(tf.greater(summary_error_diffs, 2.8), tf.ones_like(summary_error_diffs), tf.zeros_like(summary_error_diffs))
                      summary_error_diffs_uint8, _ = scale_to_255(summary_error_diffs_greater2_8, pixel_scaling=None, batch_scale=True)
                      summaries.add(tf.summary.image('metrics_map'+label_postfix+'/depth_diff_abs_error_map_greater2.8', tf.gather(summary_error_diffs_uint8, gather_list)))
              else:
                  for output_idx, output in enumerate(['u', 'v', 'z_object']):
                      summary_error_diff = tf.expand_dims(tf.gather(summary_error_diffs, output_idx, axis=3), -1)
                      summary_error_diff_uint8, _ = scale_to_255(summary_error_diff, pixel_scaling=None, batch_scale=True)
                      summaries.add(tf.summary.image('metrics_map'+label_postfix+'/%s_%s' % (error_map_name, output), tf.gather(summary_error_diff_uint8, gather_list)))

          if FLAGS.if_summary_shape_metrics:
              shape_id_sim_map_train = graph.get_tensor_by_name(pattern_train%'shape_id_sim_map')
              shape_id_sim_map_uint8_train, _ = scale_to_255(shape_id_sim_map_train, pixel_scaling=(0., 255.))
              summaries.add(tf.summary.image('metrics_map/shape_id_sim_map-trainInv', tf.gather(shape_id_sim_map_uint8_train, gather_list)))

              shape_id_sim_map = graph.get_tensor_by_name(pattern%'shape_id_sim_map')
              shape_id_sim_map_uint8, _ = scale_to_255(shape_id_sim_map, pixel_scaling=(0., 255.))
              summaries.add(tf.summary.image('metrics_map/shape_id_sim_map-valInv', tf.gather(shape_id_sim_map_uint8, gather_list)))

          if FLAGS.if_uvflow and not(FLAGS.if_depth_only):
              for appx in ['', '_flow']:
                  label_uv_map = graph.get_tensor_by_name(pattern%'label_uv%s_map'%appx)
                  logits_uv_map = graph.get_tensor_by_name(pattern%'logits_uv%s_map'%appx)
                  for output_idx, output in enumerate(['u', 'v']):
                      summary_label_output = tf.gather(label_uv_map, [output_idx], axis=3)
                      summary_label_output_uint8, pixel_scaling = scale_to_255(summary_label_output)
                      summaries.add(tf.summary.image('test'+label_postfix+'/%s%s_label' % (output, appx), tf.gather(summary_label_output_uint8, gather_list)))

                      summary_logits_output = tf.gather(logits_uv_map, [output_idx], axis=3)
                      summary_logits_output = mask_rescaled_float * summary_logits_output
                      summary_logits_output_uint8, _ = scale_to_255(summary_logits_output, pixel_scaling)
                      summaries.add(tf.summary.image('test'+label_postfix+'/%s%s_logits' % (output, appx), tf.gather(summary_logits_output_uint8, gather_list)))

          if FLAGS.if_depth_only:
              trans_hist_names = ['depth_diff_abs_error', 'depth_relative_error']
          else:
              trans_hist_names = ['trans_sqrt_error', 'depth_diff_abs_error', 'depth_relative_error', 'x_l1', 'y_l1']
          for trans_metrics in trans_hist_names:
              if pattern == pattern_val:
                summary_trans = graph.get_tensor_by_name(pattern%trans_metrics)
              else:
                summary_trans = train_utils.get_avg_tensor_from_scopes(FLAGS.num_clones, '%s:0', graph, config, trans_metrics, return_concat=True)
              if trans_metrics == 'depth_diff_abs_error':
                  summary_trans = tf.boolean_mask(summary_trans, summary_trans < 10.)
              if trans_metrics == 'depth_relative_error':
                  summary_trans = tf.boolean_mask(summary_trans, summary_trans < 1.)
              summaries.add(tf.summary.histogram('metrics_map'+label_postfix+'/%s' % trans_metrics, summary_trans))

          if pattern == pattern_val:
              depth_diff_abs_error = graph.get_tensor_by_name(pattern%'depth_diff_abs_error')
          else:
              depth_diff_abs_error = train_utils.get_avg_tensor_from_scopes(FLAGS.num_clones, '%s:0', graph, config, 'depth_diff_abs_error', return_concat=True)
          depth_diff_abs_error_thres2_8 = tf.reduce_sum(tf.to_float(tf.logical_and(depth_diff_abs_error<2.8, depth_diff_abs_error>0.))) / tf.reduce_sum(tf.to_float(depth_diff_abs_error>0.))
          summaries.add(tf.summary.scalar(('total_loss%s/'%label_postfix+pattern%'loss_reg_Zdepth_metric_thres2_8').replace(':0', ''), depth_diff_abs_error_thres2_8))

          label_outputs = graph.get_tensor_by_name(pattern%'label_pose_shape_map')
          logit_outputs = graph.get_tensor_by_name(pattern%'prob_logits_pose_shape_map')

          label_depth = graph.get_tensor_by_name(pattern%'depth_rescaled_label_map')
          label_id_depth = graph.get_tensor_by_name(pattern%'depth_rescaled_label_id_map')
          mask_depth = graph.get_tensor_by_name(pattern%'depth_rescaled_mask_map')
          logit_depth = graph.get_tensor_by_name(pattern%'depth_rescaled_logit_map')
          error_depth_cls = graph.get_tensor_by_name(pattern%'depth_rescaled_cls_error_map')
          error_depth_reg = graph.get_tensor_by_name(pattern%'depth_rescaled_reg_error_map')

          depth_log_offset_rescaled_label_map = graph.get_tensor_by_name(pattern%'depth_log_offset_rescaled_label_map')
          depth_log_offset_rescaled_logit_map = graph.get_tensor_by_name(pattern%'depth_log_offset_rescaled_logit_map')
          for output_idx, output in enumerate(dataset.output_names_summary):
              if not(FLAGS.if_shape) and 'q' in output:
                  continue
              if FLAGS.if_depth_only and output != 'z_object':
                  continue
              summary_label_output = tf.gather(label_outputs, [output_idx], axis=3)
              summary_label_output= tf.multiply(summary_mask_float_filtered, summary_label_output)
              summary_label_output_uint8, pixel_scaling = scale_to_255(summary_label_output)
              summaries.add(tf.summary.image('output'+label_postfix+'/%s_label' % output, tf.gather(summary_label_output_uint8, gather_list)))


              summary_logits_output = tf.gather(logit_outputs, [output_idx], axis=3)
              summary_logits_output = tf.multiply(summary_mask_float_filtered, summary_logits_output)
              summary_logits_output_uint8, _ = scale_to_255(summary_logits_output, pixel_scaling)
              summaries.add(tf.summary.image(
                  'output'+label_postfix+'/%s_logit' % output, tf.gather(summary_logits_output_uint8, gather_list)))

              summary_weights_output = graph.get_tensor_by_name(pattern%('%s_weights_map'%output))
              summary_weights_output = mask_rescaled_float * summary_weights_output
              summary_weights_output_uint8, _ = scale_to_255(summary_weights_output)
              summaries.add(tf.summary.image(
                  'output'+label_postfix+'/%s_weights' % output, tf.gather(summary_weights_output_uint8, gather_list)))

              if output == 'z_object': # for dense depth map
                  label_depth_output_uint8, pixel_scaling = scale_to_255(tf.multiply(mask_depth, label_depth))
                  summaries.add(tf.summary.image('dense_depth'+label_postfix+'/z_depth_dense_label', tf.gather(label_depth_output_uint8, gather_list)))

                  # label_id_depth_output_uint8, _ = scale_to_255(tf.multiply(mask_depth, tf.to_float(label_id_depth)))
                  # summaries.add(tf.summary.image('dense_depth'+label_postfix+'/%s_depth_label_id' % output, tf.gather(label_id_depth_output_uint8, gather_list)))

                  logit_depth_output_uint8, _ = scale_to_255(logit_depth, pixel_scaling)
                  summaries.add(tf.summary.image('dense_depth'+label_postfix+'/z_depth_dense_logit', tf.gather(logit_depth_output_uint8, gather_list)))

                  mask_depth_output_uint8, _ = scale_to_255(mask_depth)
                  summaries.add(tf.summary.image('dense_depth'+label_postfix+'/z_depth_mask', tf.gather(mask_depth_output_uint8, gather_list)))

                  error_depth_output_uint8, _ = scale_to_255(error_depth_cls, pixel_scaling=None, batch_scale=True)
                  summaries.add(tf.summary.image('dense_depth'+label_postfix+'/z_depth_dense_error_cls', tf.gather(error_depth_output_uint8, gather_list)))

                  error_depth_output_uint8, _ = scale_to_255(error_depth_reg, pixel_scaling=None, batch_scale=True)
                  summaries.add(tf.summary.image('dense_depth'+label_postfix+'/z_depth_dense_error_reg', tf.gather(error_depth_output_uint8, gather_list)))

                  label_depth_output_uint8, pixel_scaling = scale_to_255(tf.multiply(mask_depth, depth_log_offset_rescaled_label_map))
                  summaries.add(tf.summary.image('dense_depth'+label_postfix+'/z_depth_log_offset_label', tf.gather(label_depth_output_uint8, gather_list)))

                  logit_depth_output_uint8, _ = scale_to_255(depth_log_offset_rescaled_logit_map, pixel_scaling)
                  summaries.add(tf.summary.image('dense_depth'+label_postfix+'/z_depth_log_offset_logit', tf.gather(logit_depth_output_uint8, gather_list)))

              # summary_seg_one_hots_output = tf.gather(seg_one_hots_outputs, [output_idx], axis=3)
              # summary_seg_one_hots_output_uint8, _ = scale_to_255(summary_seg_one_hots_output, pixel_scaling=(0., 255.))
              # summaries.add(tf.summary.image('test/%s_one_hot' % output, tf.gather(summary_seg_one_hots_output_uint8, gather_list)))

              # summary_label_id_output = tf.to_float(tf.gather(label_id_outputs, [output_idx], axis=3))
              # summary_label_id_output = tf.where(summary_mask, summary_label_id_output+1, tf.zeros_like(summary_label_id_output))
              # summary_label_id_output_uint8, _ = scale_to_255(summary_label_id_output)
              # summary_label_id_output_uint8 = tf.identity(summary_label_id_output_uint8, 'tttt'+output)
              # summaries.add(tf.summary.image(
              #     'test/%s_label_id' % output, tf.gather(summary_label_id_output_uint8, gather_list)))

              summary_diff, _ = scale_to_255(tf.abs(tf.to_float(summary_label_output_uint8) - tf.to_float(summary_logits_output_uint8)))
              summary_diff = tf.multiply(summary_mask_float_filtered, tf.to_float(summary_diff))
              summaries.add(tf.summary.image('diff_map'+label_postfix+'/%s_ldiff' % output, tf.gather(tf.cast(summary_diff, tf.uint8), gather_list)))

              # if output_idx in [0, 1, 2, 3, 4,5,6]:
              #     summary_loss_slice_reg = graph.get_tensor_by_name((pattern%'loss_slice_reg_').replace(':0', '')+output+':0')
              #     summaries.add(tf.summary.scalar('slice_loss'+label_postfix+'/'+(pattern%'reg_').replace(':0', '')+output, summary_loss_slice_reg))

              #     if output_idx in [0, 1, 2, 3, 6]:
              #         summary_loss = graph.get_tensor_by_name((pattern%'loss_slice_cls_').replace(':0', '')+output+':0')
              #         summaries.add(tf.summary.scalar('slice_loss'+label_postfix+'/'+(pattern%'cls_').replace(':0', '')+output, summary_loss))

          add_shape_metrics = ['loss_all_shape_id_cls_metric', 'loss_reg_shape'] if FLAGS.if_summary_shape_metrics else []
          add_uv_metrics = ['loss_reg_uv_map'] if (FLAGS.if_uvflow and not(FLAGS.if_depth_only)) else []
          add_trans_metrics = ['loss_reg_Zdepth_metric', 'loss_reg_Zdepth_relative_metric', 'loss_reg_trans', 'loss_slice_cls_dense_z', 'loss_slice_l1_offset_z'] if FLAGS.if_depth_only else ['loss_reg_rot_quat_metric', 'loss_reg_rot_quat', 'loss_reg_trans_metric', 'loss_reg_Zdepth_metric', 'loss_reg_Zdepth_relative_metric', 'loss_reg_x_metric', 'loss_reg_y_metric', 'loss_reg_trans']
          for loss_name in ['loss_cls_ALL'] + add_shape_metrics + add_uv_metrics + add_trans_metrics:
              if pattern == pattern_val:
                summary_loss_avg = graph.get_tensor_by_name(pattern%loss_name)
              else:
                summary_loss_avg = train_utils.get_avg_tensor_from_scopes(FLAGS.num_clones, '%s:0', graph, config, loss_name)
              summaries.add(tf.summary.scalar(('total_loss%s/'%label_postfix+pattern%loss_name).replace(':0', ''), summary_loss_avg))

    summaries |= set(
        tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

    return summaries
