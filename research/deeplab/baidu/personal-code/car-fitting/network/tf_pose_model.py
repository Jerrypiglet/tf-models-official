import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net


def placeholder_inputs(batch_size, num_point, feat_dim=3):
    pointclouds_pl = tf.placeholder(tf.float32,
                         shape=(batch_size, num_point, feat_dim))
    labels_pl = tf.placeholder(tf.int32,
                         shape=(batch_size, num_point, feat_dim))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx6, output BxNx6 """

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    feat_dim = point_cloud.get_shape()[2].value
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=feat_dim)

    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,feat_dim],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)

    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    point_feat = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(point_feat, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point, 1],
                         padding='VALID', scope='maxpool')
    print(global_feat)

    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    concat_feat = tf.concat([point_feat, global_feat_expand], 3)
    print(concat_feat)

    net = tf_util.conv2d(concat_feat, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)

    net = tf_util.conv2d(net, feat_dim, [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10')
    net = tf.squeeze(net, [2]) # BxNxfeat_dim
    net = net + point_cloud

    return net, end_points


def get_pose_loss(pred, label,  end_points, balance=100., reg_weight=0.001):
    def slice_pose(pose_in):
        trans = tf.slice(pose_in, [0, 0, 3], [-1, -1, 3])
        rot = tf.slice(pose_in, [0, 0, 0], [-1, -1, 3])
        return trans, rot

    pred_trans, pred_rot = slice_pose(pred)
    label_trans, label_rot = slice_pose(label)

    loss_trans = tf.losses.huber_loss(
            label_trans, pred_trans,
            weights=1.0, delta=5.0)
    loss_rot = tf.losses.huber_loss(
            label_rot, pred_rot,
            weights=1.0, delta=1.0)

    pose_loss = loss_rot * balance + loss_trans
    tf.summary.scalar('pose loss', pose_loss)

    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat_loss', mat_diff_loss)

    return pose_loss + mat_diff_loss * reg_weight


def get_loss(pred, label, end_points, reg_weight=0.001):
    """  pred: BxNxC,
        label: BxN,
    """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat_loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
