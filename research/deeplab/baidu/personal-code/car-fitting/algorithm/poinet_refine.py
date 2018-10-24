import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import pdb

# sys.path.append(os.path.join(BASE_DIR, 'network'))
import data.data_iter as data_iter
import data.kitti as kitti
import data.apolloscape as apollo
import data.kitti_env as kitti_env
import data.apolloscape_env as apollo_env
import utils.metric as eval_metric
import data.data_setting as setting

data_libs = {}
data_libs['kitti'] = kitti
data_libs['apollo'] = apollo
data_libs['kitti_env'] = kitti_env
data_libs['apollo_env'] = apollo_env

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='network.tf_pose_model', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log/tf_model/', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=32, help='Point Number [32/64/128/256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
# MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
# os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
# os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train(dataset='apollo'):
    import config.policy_config_apollo as config

    params = data_libs[dataset].set_params_disp(disp='psm')
    data_setting, label_setting = setting.get_gnn_data_setting(params, FLAGS)
    train_iter = data_iter.GNNIter(params=params,
            config=FLAGS,
            set_name='train',
            data_type='np',
            data_setting=data_setting,
            label_setting=label_setting)

    val_iter = data_iter.GNNIter(params=params,
            config=FLAGS,
            set_name='val',
            data_type='np',
            data_setting=data_setting,
            label_setting=label_setting)

    pose_eval_metric = eval_metric.PoseMetric(
            is_euler=True,
            trans_idx=[3, 4, 5], rot_idx=[0, 1, 2],
            data_type=eval_metric.InputType.NUMPY)

    pose_eval_metric.reset()
    feat_dim = 6

    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE,
                    NUM_POINT, feat_dim=feat_dim)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.get_model(
                    pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_pose_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(
                        learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer, train_iter,
                    pose_eval_metric, epoch)
            eval_one_epoch(sess, ops, test_writer, val_iter,
                    pose_eval_metric, epoch)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer, data_iter, pose_metric, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    for nbatch, data_batch in enumerate(data_iter):
        # data_batch.data[0], batch_size x n_cars x 6
        # data_batch.label[0], batch_size x n_cars x 6

        # Augment batched point clouds by rotation and jittering
        feed_dict = {ops['pointclouds_pl']: data_batch.data[0],
                     ops['labels_pl']: data_batch.label[0],
                     ops['is_training_pl']: is_training}

        summary, step, _, loss_val, pred_val = sess.run(
                [ops['merged'], ops['step'],
                 ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pose_metric.update(data_batch.label[0], pred_val)

        if nbatch % 10 == 0:
            eval_name_vals = pose_metric.get()
            log_str = "Epoch[%d] Batch[%d] " % (epoch, nbatch)
            for name, val in zip(eval_name_vals[0], eval_name_vals[1]):
                log_str = log_str + "%s=%f " % (name, val)
            logging.info(log_str)

    data_iter.reset()
    eval_name_vals = pose_metric.get()
    for name, val in zip(eval_name_vals[0], eval_name_vals[1]):
        logging.info("Epoch[%d] Train-%s=%f" % (epoch, name, val))
    pose_metric.reset()


def eval_one_epoch(sess, ops, test_writer, data_iter, pose_metric, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    # Shuffle train files
    for nbatch, data_batch in enumerate(data_iter):
        feed_dict = {ops['pointclouds_pl']: data_batch.data[0],
                     ops['labels_pl']: data_batch.label[0],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        pose_metric.update(data_batch.label[0], pred_val)

    data_iter.reset()
    eval_name_vals = pose_metric.get()
    for name, val in zip(eval_name_vals[0], eval_name_vals[1]):
        logging.info("Epoch[%d] Validation-%s=%f" % (epoch, name, val))
    pose_metric.reset()




if __name__ == "__main__":
    train()
    LOG_FOUT.close()
