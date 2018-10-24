import cv2
import os
import utils.metric as eval_metric
import argparse
import time
import numpy as np
import algorithm.ddpg_car_fit as car_fit
import algorithm.pg_car_fit as pg_car_fit
import data.data_setting as data_setting
import logging
import pdb
import utils.utils as uts
import network.car_pose_net as pose_net
from collections import namedtuple
import evaluation.eval_utils as eval_uts

import data.kitti as kitti
import data.kitti_iter as data_iter
import data.apolloscape as apollo
import data.kitti_env as kitti_env
import data.apolloscape_env as apollo_env
import mxnet as mx

import Networks.net_util as nuts
import pickle as pkl

data_libs = {}
data_libs['kitti'] = kitti
data_libs['apollo'] = apollo
data_libs['kitti_env'] = kitti_env
data_libs['apollo_env'] = apollo_env
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def overwrite_config(c, args):
    c.dataset = args.dataset
    c.gpu_id = args.gpu_id
    c.network = args.network
    c.w_agent = True
    c.is_crop = True
    c.is_rel = False # if true the network output is absolute pose


def eval_with_data_iter(args):

    Batch = namedtuple('Batch', ['data'])
    dataset = args.dataset
    if args.dataset == 'kitti':
        import config.policy_config as config
    elif args.dataset == 'apollo':
        import config.policy_config_apollo as config
    else:
        raise ValueError(' no such dataset ')
    overwrite_config(config, args)

    params = data_libs[args.dataset].set_params_disp()
    params['batch_size'] = 1
    params['crop_size'] = [config.height, config.width]

    val_set = 'minival' if dataset == 'apollo' else 'val'
    env_eval = data_libs[dataset + '_env'].Env(config, params, split=val_set)
    logging.info('init data iter val')
    sampler_val = env_eval.sample_state_rect if dataset == 'apollo' else \
            env_eval.sample_mcmc_state
    setting = data_setting.get_policy_data_setting(env_eval)
    data_names = ['image', 'depth', 'mask', 'crop', 'pose']
    label_names = ['del_pose']

    val_iter = data_iter.PolicyDataIter(
        params=params, env=env_eval,
        sampler=sampler_val,
        setting=setting,
        data_names=data_names,
        label_names=label_names)
    val_iter.reset()

    pose_eval_metric = eval_metric.PoseMetric(
            output_names=['pose_output'],
            label_names=env_eval.action_names, is_euler=True,
            trans_idx=[3, 4, 5], rot_idx=[0, 1, 2])
    pose_eval_metric.reset()

    state = nuts.get_mx_var_by_name(data_names)
    act_sym = pose_net.pose_block_w_crop(state, params, name='pose',
             is_rel=True)

    data_shapes = [v for k, v in val_iter.provide_data]
    in_model = {'model':args.model_name}
    model = nuts.load_model(in_model,
                 data_names=data_names,
                 data_shapes=data_shapes,
                 net=act_sym['del_pose'], ctx=mx.gpu(0))

    for nbatch, eval_batch in enumerate(val_iter):
        # if nbatch % 10 == 0:
        #     print nbatch
        model.forward(Batch(eval_batch.data))
        output_nd = model.get_outputs()
        pose_eval_metric.update(eval_batch.label, output_nd)

    epoch = int(args.model_name.split('-')[-1])
    eval_name_vals = pose_eval_metric.get()
    for name, val in zip(eval_name_vals[0], eval_name_vals[1]):
        logging.info("Epoch[%d] Validation-%s=%f" % (epoch, name, val))


def mp_eval(args):
    if args.dataset == 'kitti':
        import config.policy_config as config
    elif args.dataset == 'apollo':
        import config.policy_config_apollo as config
    else:
        raise ValueError(' no such dataset ')

    overwrite_config(config, args)
    # print [(item, eval('config.' + item)) for item in dir(config) \
    #         if not item.startswith("__")]

    params = data_libs[args.dataset].set_params_disp()
    env = data_libs[args.dataset + '_env'].Env(config, params, split='minival')
    logging.info('Evaluate model %s' % args.model_name)

    models = {'actor': args.model_name}
    agent = pg_car_fit.PGCarFitting(config, env)
    agent.pgnet.init_params(models)

    total_reward, total_IOU, total_delta = [0.0 for i in range(3)]
    counter = 0
    val_set_name = 'minival' if args.dataset == 'apollo' else 'val'
    env_eval = data_libs[args.dataset + '_env'].Env(config,
            params, split=val_set_name)
    sampler_eval = env_eval.sample_mcmc_state if args.dataset == 'kitti' else \
            env_eval.sample_state_rect
    pose_eval_metric = eval_metric.PoseMetric(
            output_names=['pose_output'],
            label_names=env_eval.action_names, is_euler=True,
            trans_idx=[3, 4, 5], rot_idx=[0, 1, 2])

    data_transform = data_setting.get_policy_data_setting(env_eval)
    car_poses = []
    while True:
        state, act = sampler_eval()
        if env_eval.counter % 100 == 0:
            logging.info('Evaluation %d / %d' % (env_eval.counter, env_eval.image_num))

        if env_eval.counter == env_eval.image_num and \
            env_eval.inst_counter == len(env_eval.valid_id):
            break

        if env_eval.inst_counter == len(env_eval.valid_id) and args.is_save:
            save_path = params['output_path'] + args.model_name.split('/')[-1]
            uts.mkdir_if_need(save_path)
            file_name = save_path + '/%s.pkl' % env_eval.image_name
            pkl.dump(car_poses, open(file_name, 'wb'))
            car_poses = []

        state_data = data_setting.data_transform_4_network(
                state, agent.state_names, data_transform)
        action = agent.self_action(state_data)
        target_pose = act['del_pose']

        pose_eval_metric.update(
                [mx.nd.array(target_pose)],
                [mx.nd.array(action['del_pose'] + state['pose'])])

        # action, reward, inspect, done = agent.mcmc_sample(
        #       action['del_pose'], state)
        # reward, done, next_state, inspect = env_eval.step(action, state)
        # max_reward, max_IOU, max_delta = eval_uts.get_max_reward(
        #         reward, inspect)
        # name = agent.replay_buffer.state_to_key(state_data)
        # total_reward += max_reward
        # total_IOU += max_IOU
        # total_delta += max_delta
        counter += 1

    epoch = int(args.model_name.split('-')[-1])
    eval_name_vals = pose_eval_metric.get()
    for name, val in zip(eval_name_vals[0], eval_name_vals[1]):
        logging.info("Epoch[%d] Validation-%s=%f" % (epoch, name, val))
    pose_eval_metric.reset()

    # logging.info("Epoch[%d] Evaluation Reward: Ave-reward %f Ave-IOU %f \
    #         Ave-delta %f" % (epoch, total_reward / counter, \
    #         total_IOU / counter, total_delta / counter))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a mx model.')
    parser.add_argument('--model_name', default='./output/policy-apollo-rel-stereo-0117',
            help='the prefix for saved model')
    parser.add_argument('--gpu_id', type=int, default=0,
        help='gpu_id to use')
    parser.add_argument('--network', type=str, default='demon',
        help='network use for training')
    parser.add_argument('--dataset', default='apollo',
        help='the data set need to train')
    parser.add_argument('--is_save', type=uts.str2bool, default='false',
        help='means we want to save prediction images.')

    args = parser.parse_args()
    mp_eval(args)
    eval_with_data_iter(args)



