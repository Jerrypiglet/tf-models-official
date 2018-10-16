import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../'))
import tf_util
from transform_nets import input_transform_net

def get_model(point_cloud, input_label, is_training, cat_num, part_num, \
    batch_size, num_point, weight_decay, bn_decay=None):

  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  input_image = tf.expand_dims(point_cloud, -1)

  k = 5
  adj = tf_util.pairwise_distance(point_cloud)
  nn_idx = tf_util.knn(adj, k=k)
  print '++++', input_image.get_shape()
  edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, squeeze_axis=3, k=k)

  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(edge_feature, is_training, bn_decay, K=point_cloud.get_shape()[2].value, is_dist=True)
  point_cloud_transformed = tf.matmul(point_cloud, transform)
  input_image = tf.expand_dims(point_cloud_transformed, -1)
  adj = tf_util.pairwise_distance(point_cloud_transformed)
  nn_idx = tf_util.knn(adj, k=k)
  print '++++', input_image.get_shape()
  edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, squeeze_axis=3, k=k)

  out1 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv1', bn_decay=bn_decay, is_dist=True)

  out2 = tf_util.conv2d(out1, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv2', bn_decay=bn_decay, is_dist=True)

  net_max_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)
  net_mean_1 = tf.reduce_mean(out2, axis=-2, keep_dims=True)

  out3 = tf_util.conv2d(tf.concat([net_max_1, net_mean_1], axis=-1), 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv3', bn_decay=bn_decay, is_dist=True)

  adj = tf_util.pairwise_distance(tf.squeeze(out3, axis=-2))
  nn_idx = tf_util.knn(adj, k=k)
  print '++++', out3.get_shape()
  edge_feature = tf_util.get_edge_feature(out3, nn_idx=nn_idx, squeeze_axis=2, k=k)

  out4 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv4', bn_decay=bn_decay, is_dist=True)

  net_max_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)
  net_mean_2 = tf.reduce_mean(out4, axis=-2, keep_dims=True)

  out5 = tf_util.conv2d(tf.concat([net_max_2, net_mean_2], axis=-1), 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv5', bn_decay=bn_decay, is_dist=True)

  adj = tf_util.pairwise_distance(tf.squeeze(out5, axis=-2))
  nn_idx = tf_util.knn(adj, k=k)
  print '++++', out5.get_shape()

