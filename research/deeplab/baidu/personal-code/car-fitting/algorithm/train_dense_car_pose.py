""" Training script for pose regression
"""
import sys
import argparse
import cv2
import mxnet as mx
import Networks.net_util as net_util
import network.car_pose_net as pose_net
import network.network_updater as updater
import Networks.mx_losses as losses
import utils.transforms as trs
import utils.utils as uts
import utils.metric as eval_metric
import logging
import pdb

import numpy as np
import config.policy_config as c
import ou_noise as noise
from collections import OrderedDict

np.set_printoptions(precision=3, suppress=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import data.data_setting as data_setting
import data.kitti as kitti
import data.apolloscape as apollo
import data.kitti_env as kitti_env
import data.apolloscape_env as apollo_env
import data.kitti_iter as data_iter

data_libs = {}
data_libs['kitti'] = kitti
data_libs['apollo'] = apollo
data_libs['kitti_env'] = kitti_env
data_libs['apollo_env'] = apollo_env


def train_iter(dataset='kitti'):
    import config.policy_config_apollo as c

    c.is_crop = args.is_crop
    c.is_rel = args.is_rel
    ctx = [mx.gpu(int(i)) for i in args.gpu_id.split(',')]
    params = data_libs[dataset].set_params_disp(disp='psm')

    # wrap the environment to
    env = data_libs[dataset + '_env'].Env_s(c, params, split='train')

    params['batch_size'] = c.batch_size
    params['size'] = env.image_size
    params['crop_size'] = [c.height, c.width]

    data_names = ['image']
    label_names = ['pose', 'shape']

    setting = data_setting.get_policy_data_setting(env)
    sampler = env.sample_image if dataset == 'kitti' else \
                        env.sample_pose_image

    logging.info('init data iter train')
    train_data_iter = data_iter.PolicyDataIter(
                params=params, env=env,
                sampler=sampler,
                setting=setting,
                data_names=data_names,
                label_names=label_names)

    val_set = 'val'
    env_eval = data_libs[dataset + '_env'].Env_s(c, params, split=val_set)
    logging.info('init data iter val')
    sampler_val = env_eval.sample_pose_image
    val_iter = data_iter.PolicyDataIter(
                params=params, env=env_eval,
                sampler=sampler_val,
                setting=setting,
                data_names=data_names,
                label_names=label_names)

    print 'Init iter'
    checkpoint = mx.callback.do_checkpoint(args.prefix)
    log = mx.callback.Speedometer(1, 10)

    state = net_util.get_mx_var_by_name(data_names)
    action = net_util.get_mx_var_by_name(label_names)

    net_name = 'pose'
    outputs = []
    # use the geometric projection loss
    sym = segnet.segnet(state, params, name=net_name,
                        is_rel=args.is_rel)

    loss = losses.my_dense_pose_loss(act_sym[env.action_names[0]],
                        action[env.action_names[0]],
                        balance=10,
                        batch_size=c.batch_size)
    outputs.append(loss)

    # have pose output for evaluation
    outputs.append(
            mx.sym.BlockGrad(act_sym[env.action_names[0]], name='pose'))
    loss = mx.symbol.Group(outputs)

    logging.info('Init network %s' % net_name)
    arg_params = None
    aux_params = None
    if not (args.init_model is None):
        print 'loading model from existing network '
        models = {'model': args.init_model}
        symbol, arg_params, aux_params = net_util.args_from_models(models, False)

    def toc_value(x):
        return mx.nd.array(x)

    obs_output = 'pose_pose_output'
    # mon = mx.monitor.Monitor(1, pattern=obs_output)
    mon = mx.monitor.Monitor(1, stat_func=toc_value, pattern='pose_0|pose_pose_output|del_pose')
    mod = mx.mod.Module(symbol=loss,
                        context=ctx,
                        data_names=data_names,
                        label_names=label_names)

    allow_missing = True
    optimizer_params = {'learning_rate':c.actor_lr,
                        'wd':c.actor_wd}

    pose_eval_metric = eval_metric.PoseMetric(
            output_names=['pose_output'],
            label_names=env.action_names, is_euler=True,
            trans_idx=[3, 4, 5], rot_idx=[0, 1, 2])
    # segment_eval_metric = eval_metric.SegMetric()

    # training
    mod.fit(train_data_iter,
            eval_data=val_iter,
            # monitor=mon,
            optimizer=c.actor_updater,
            eval_metric=[pose_eval_metric, 'loss'],
            optimizer_params=optimizer_params,
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=allow_missing,
            num_epoch=c.epoch,
            kvstore = mx.kvstore.create(args.kv_store),
            batch_end_callback=log,
            epoch_end_callback=checkpoint)

def train_resnet_iter_descrete(dataset, network='resnet'):
    if dataset == 'kitti':
        import config.policy_config as c
    elif dataset == 'apollo':
        import config.policy_config_apollo as c

    c.is_crop = args.is_crop
    c.is_rel = args.is_rel
    ctx = [mx.gpu(int(i)) for i in args.gpu_id.split(',')]
    params = data_libs[dataset].set_params_disp(disp='psm')

    # relative pose range
    pose_range = [[-np.pi/4, np.pi/4],
                  [-np.pi/4, np.pi/4],
                  [-np.pi, np.pi],
                  [-5.0, 10.0],
                  [-5.0, 10.0],
                  [-10.0, 10.0]]
    bin_nums = [8, 8, 64, 16, 16, 64]
    bin_range = [np.linspace(r[0], r[1], num=b).tolist() \
            for r, b in zip(pose_range, bin_nums)]

    # wrap the environment to
    params['bins'] = bin_range
    env = data_libs[dataset + '_env'].Env(c, params, split='train')

    params['batch_size'] = c.batch_size
    params['size'] = env.image_size
    params['crop_size'] = [c.height, c.width]
    data_names = ['image', 'depth', 'mask', 'crop', 'pose']
    label_names = ['del_pose', 'disc_pose']
    space_names = ['row', 'pitch', 'yaw', 'x', 'y', 'z']

    setting = data_setting.get_policy_data_setting(env)
    if args.is_sim:
        sampler = env.simulate_state
    else:
        sampler = env.sample_mcmc_state \
                if dataset == 'kitti' else \
                env.sample_state_rect

    logging.info('init data iter train')
    train_iter = data_iter.PolicyDataIter(
            params=params, env=env,
            sampler=sampler,
            setting=setting,
            data_names=data_names,
            label_names=label_names)

    val_set = 'minival' if dataset == 'apollo' else 'val'
    env_eval = data_libs[dataset + '_env'].Env(c, params, split=val_set)
    logging.info('init data iter val')
    sampler_val = env_eval.sample_state_rect if dataset == 'apollo' else \
            env_eval.sample_mcmc_state

    val_iter = data_iter.PolicyDataIter(
        params=params, env=env_eval,
        sampler=sampler_val,
        setting=setting,
        data_names=data_names,
        label_names=label_names)

    print 'Init iter'
    checkpoint = mx.callback.do_checkpoint(args.prefix)
    log = mx.callback.Speedometer(1, 10)

    state = net_util.get_mx_var_by_name(data_names)
    action = net_util.get_mx_var_by_name(label_names)

    bin_vals = [mx.sym.Variable(name, shape=(1, bin_nums[i]),
                init=net_util.MXConstant(value=[bin_range[i]])) \
                for i, name in enumerate(space_names)]
    bin_vals = [mx.sym.BlockGrad(bin_val) for bin_val in bin_vals]

    net_name = 'actor'
    # use the geometric projection loss
    act_sym = pose_net.resnext_pose_block(state, params,
            name=net_name, is_rel=args.is_rel, is_discrete=True,
            bin_nums=bin_nums, bin_vals=bin_vals, bin_names=space_names)

    outputs = []
    loss = losses.my_disc_pose_loss(act_sym,
                                    action,
                                    batch_size=c.batch_size,
                                    bin_nums=bin_nums)
    if not isinstance(loss, list):
        outputs.append(loss)
    else:
        outputs = loss

    # have pose output for evaluation
    outputs.append(mx.sym.BlockGrad(act_sym[env.action_names[0]],
                   name='pose'))
    loss = mx.symbol.Group(outputs)

    logging.info('Init network %s' % net_name)
    arg_params = None
    aux_params = None

    print 'loading model from existing network '
    models = {'model': args.init_model}
    symbol, arg_params, aux_params = net_util.args_from_models(models,
            with_symbol=False)

    mon = mx.monitor.Monitor(1)
    mod = mx.mod.Module(symbol=loss,
                        context=ctx,
                        data_names=data_names,
                        label_names=label_names)

    allow_missing = True
    optimizer_params = {'learning_rate': c.actor_lr,
                        'wd':c.actor_wd}

    pose_eval_metric = eval_metric.PoseMetric(
            output_names=['pose_output'],
            label_names=env.action_names, is_euler=True,
            trans_idx=[3, 4, 5], rot_idx=[0, 1, 2])

    mod.fit(train_iter,
            eval_data=val_iter,
            # monitor=mon,
            optimizer=c.actor_updater,
            eval_metric=[pose_eval_metric, 'loss'],
            optimizer_params=optimizer_params,
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=allow_missing,
            num_epoch=c.epoch,
            kvstore = mx.kvstore.create(args.kv_store),
            batch_end_callback=log,
            epoch_end_callback=checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a mx model.')
    parser.add_argument('--init_model', default=None,
        help='The type of fcn-xs model for init, e.g. fcnxs, fcn16s, fcn8s.')
    parser.add_argument('--gpu_id', default="2,3", help='the gpu ids use for training')
    parser.add_argument('--dataset', default="apollo",
        help='the dataset use for training')
    parser.add_argument('--prefix', default='./output/policy-net',
        help='policy net')
    parser.add_argument('--is_pg', type=uts.str2bool, default='false',
        help='true means we use policy gradient for training.')
    parser.add_argument('--is_sim', type=uts.str2bool, default='false',
        help='true means we use simulated data.')
    parser.add_argument('--is_rel', type=uts.str2bool, default='true',
        help='true means we train relative pose.')
    parser.add_argument('--is_crop', type=uts.str2bool, default='true',
        help='true means we train relative pose.')
    parser.add_argument('--is_discret', type=uts.str2bool, default='false',
        help='true means we train relative pose.')
    parser.add_argument('--network', default='resnext',
        help='policy net')
    parser.add_argument('--kv_store', type=str, default='device', help='the kvstore type')
    args = parser.parse_args()
    logging.info(args)

    if args.network == 'resnet' or args.network == 'resnext':
            train_resnet_iter_descrete(args.dataset, args.network)
        else:
            train_resnet_iter(args.dataset, args.network)
    elif args.network == 'demon':
        train_iter(args.dataset)
    else:
        raise ValueError(' No such network ')


