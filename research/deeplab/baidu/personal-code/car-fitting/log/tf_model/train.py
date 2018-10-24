from __future__ import division
import os
import utils.metric as eval_metric
import argparse
import mxnet as mx
import pprint
import time
import numpy as np
import algorithm.ddpg_car_fit as car_fit
import algorithm.pg_car_fit as pg_car_fit
import data.data_setting as data_setting
import logging
import pdb
import utils.utils as uts
from collections import OrderedDict
import evaluation.eval_utils as eval_uts

import data.kitti as kitti
import data.apolloscape as apollo
import data.kitti_env as kitti_env
import data.apolloscape_env as apollo_env
import data.kitti_iter as data_iter
import pickle as pkl

data_libs = {}
data_libs['kitti'] = kitti
data_libs['apollo'] = apollo
data_libs['kitti_env'] = kitti_env
data_libs['apollo_env'] = apollo_env
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
# print os.environ['MXNET_ENGINE_TYPE']

def get_max_reward(reward, inspect):
    max_reward = np.max(reward['reward'])
    idx = np.argmax(reward['reward'])
    max_IOU = inspect['IOU'][idx]
    max_delta = inspect['delta'][idx]

    return max_reward, max_IOU, max_delta


def train_agent(data_set='kitti'):
    import config.config as c

    params = kitti.set_params_disp()
    env = eval(data_set + '_env.Env(c, params)')
    rand_init = True
    setting = data_setting.get_policy_data_setting(env)
    agent = car_fit.DDPGCarFitting(env, c)
    models = {'actor':args.init_actor_model,
              'critic':args.init_critic_model}
    agent.load_networks(models)

    pose_eval_metric = eval_metric.PoseMetric(
            output_names=['pose_output'],
            label_names=env.action_names, is_euler=True,
            trans_idx=[3, 4, 5], rot_idx=[0, 1, 2])

    for episode in xrange(c.EPISODES):
        # sample a car instance
        state, gt_action = env.simulate_state_from_image(
                is_rand=rand_init)
        state_data = OrderedDict({})
        for name in agent.ddpgnet.actor_in_names:
            state_data[name] = setting[name]['transform'](
                state[name].copy())

        # Train
        max_reward, max_IOU, max_delta  = [0.0 for i in range(3)]
        for step in xrange(c.timestep_limit):
            # transform state to be input of network
            action = agent.noise_action(state_data)
            reward, done, next_state, inspect = env.step(action)
            agent.perceive(state_data, action, reward, next_state, done)
            max_reward = max(max_reward, np.max(reward['reward']))
            idx = np.argmax(reward['reward'])
            max_IOU = max(max_IOU, inspect['IOU'][idx])
            max_delta = max(max_delta, inspect['delta'][idx])
            if done:
                break

        # maximum sampled reward
        logging.info("episode: %d Evaluation Reward: %f max IOU %f, depth max rel %f" % (episode, max_reward, max_IOU, max_delta))

        # Validation:
        if episode % c.eval_iter == 0:
            total_reward, total_IOU, total_delta = [0, 0, 0]
            for i in xrange(c.TEST):
                # state = env.sample_state()
                state, act = env.simulate_state_from_image(
                        is_rand=rand_init)
                state_data = OrderedDict({})
                for name in agent.ddpgnet.actor_in_names:
                    state_data[name] = setting[name]['transform'](
                            state[name].copy())

                action = agent.self_action(state_data)
                reward, done, next_state, inspect = env.step(action)
                max_reward, max_IOU, max_delta = get_max_reward(reward,
                        inspect)
                pose_eval_metric.update(
                        [mx.nd.array(act['del_pose'])],
                        [mx.nd.array(action['del_pose'])])

                total_reward += max_reward
                total_IOU += max_IOU
                total_delta += max_delta

            logging.info("Epoch[%d] Evaluation Reward: \
                   Ave-reward %f Ave-IOU %f Ave-delta %f" % (
                   episode, total_reward / c.TEST, total_IOU / c.TEST,
                   total_delta / c.TEST))

            eval_name_vals = pose_eval_metric.get()
            for name, val in zip(eval_name_vals[0], eval_name_vals[1]):
                logging.info("Epoch[%d] Validation-%s=%f" % (episode,
                   name, val))
            pose_eval_metric.reset()
            agent.save_networks(episode / c.eval_iter)


def data_transform_4_network(state, state_names, data_transform):
    state_data = OrderedDict({})

    # for putting into hashing table
    state_data['image_name'] = state['image_name']
    state_data['inst_id'] = state['inst_id']
    for name in state_names:
        trans = data_transform[name]['transform']
        params = data_transform[name]['params'] if 'params' in \
                data_transform[name].keys() else {}
        if 'depth' in name:
            params = {'mean_depth': state['pose'][0, -1]}
        state_data[name] = trans(state[name], **params)
    return state_data


def train_policy_agent(data='kitti', with_agent=True):
    if data == 'kitti':
        import config.policy_config as c
    elif data == 'apollo':
        import config.policy_config_apollo as c
    else:
        raise ValueError(' no such dataset ')

    # map the args to configuration
    c.gpu_id = args.gpu_id
    c.prefix = args.prefix
    c.network = args.network
    c.is_crop = True
    c.is_rel = args.is_rel # if true the network output is absolute pose
    c.has_network = with_agent
    print [(item, eval('c.' + item)) for item in dir(c) if not item.startswith("__")]

    params = data_libs[data].set_params_disp()
    env = data_libs[data + '_env'].Env(c, params, split='train')
    sampler_train = env.sample_mcmc_state if data == 'kitti' else env.sample_state_rect

    models = {'actor': args.init_actor_model}
    agent = pg_car_fit.PGCarFitting(c, env)
    agent.pgnet.init_params(models)

    data_transform = data_setting.get_policy_data_setting(env)
    pose_eval_metric = eval_metric.PoseMetric(
            output_names=['pose_output'],
            label_names=env.action_names, is_euler=True,
            trans_idx=[3, 4, 5], rot_idx=[0, 1, 2])
    Epoch = 0

    for episode in xrange(c.EPISODES):
        # sample a car instance
        state, act = sampler_train()

        # state, act = env.simulate_state_from_image(is_rand=rand_init)
        state_data = data_transform_4_network(state, agent.state_names, data_transform)
        max_reward, max_IOU, max_delta  = [-1 * np.inf for i in range(3)]

        # transform state to be input of network
        action = agent.self_action(state_data)
        s = time.time()
        action, reward, inspect, done, is_better = agent.mcmc_sample(
                            action, state)
        if not is_better:
            continue

        sample_time = time.time() - s
        max_reward, max_IOU, max_delta = eval_uts.get_max_reward(
                reward, inspect, [max_reward, max_IOU, max_delta])
        reward_in = {'reward': max_reward * np.ones((1, 1))}
        agent.perceive(state_data, action, reward=reward_in, done=done)

        # maximum sampled reward
        logging.info("Image [%s] Car Inst [%d] Reward: %f, max IOU: %f, depth max rel: %f,  time cost: %f " \
                % (env.image_name, env.inst_counter, max_reward, max_IOU, max_delta, \
                sample_time))

        # Validation:
        # if episode % env.image_num == 0 and episode > env.image_num:
        if episode == 0 or (env.counter == env.image_num and \
                            env.inst_counter == len(env.valid_id)):
            total_reward, total_IOU, total_delta = [0.0 for i in range(3)]
            counter = 0
            val_set_name = 'minival' if data == 'apollo' else 'val'
            env_eval = data_libs[data + '_env'].Env(c, params, split=val_set_name)
            sampler_eval = env_eval.sample_mcmc_state if data == 'kitti' else \
                    env_eval.sample_state_rect
            while True:
                state, act = sampler_eval()
                # state, act = env_eval.simulate_state_from_image(is_rand=rand_init)
                if env_eval.counter == env_eval.image_num and \
                    env_eval.inst_counter == len(env_eval.valid_id):
                    break
                state_data = data_transform_4_network(
                      state, agent.state_names, data_transform)
                action = agent.self_action(state_data)
                # action = {'del_pose': np.zeros((1, 6))}
                # action, reward, inspect, done = agent.mcmc_sample(
                #         action['del_pose'], state)
                reward, done, next_state, inspect = env_eval.step(action, state)
                max_reward, max_IOU, max_delta = eval_uts.get_max_reward(
                        reward, inspect)
                name = agent.replay_buffer.state_to_key(state_data)
                if name in agent.replay_buffer.buffer.keys():
                    buffer_del_pose = agent.replay_buffer.buffer[name]['del_pose']
                    target_pose = buffer_del_pose + state['pose']
                else:
                    target_pose = act['del_pose']

                pose_eval_metric.update(
                        [mx.nd.array(target_pose)],
                        [mx.nd.array(action['del_pose'] + state['pose'])])

                total_reward += max_reward
                total_IOU += max_IOU
                total_delta += max_delta
                counter += 1

            logging.info("Epoch[%d] Evaluation Reward: Ave-reward %f Ave-IOU %f \
                    Ave-delta %f" % (Epoch, total_reward / counter, \
                    total_IOU / counter, total_delta / counter))

            eval_name_vals = pose_eval_metric.get()
            for name, val in zip(eval_name_vals[0], eval_name_vals[1]):
                logging.info("Epoch[%d] Validation-%s=%f" % (episode,
                    name, val))
            pose_eval_metric.reset()
            agent.save_networks(Epoch)
            Epoch += 1

        # if episode % 50 == 0:
        #     max_reward, max_IOU, max_delta  = [0.0 for i in range(3)]
        #     test_num = 100
        #     for i in xrange(test_num):
        #         state_test= env_eval.sample_state()
        #         action = agent.self_action(state_test)
        #         reward, _, _, inspector = env_eval.step(action, state_test)
        #         max_reward += np.max(reward['reward'])
        #         max_IOU += np.max(inspector['IOU'])
        #         max_delta += np.max(inspector['delta'])
        #     logging.info("Testing episode: %d Reward: %f max IOU %f, depth max rel %f" % (episode, max_reward/test_num, max_IOU/test_num, max_delta/test_num))


def mcmc_policy_agent(data='kitti', with_agent=False):

    if data == 'kitti':
        import config.policy_config as c
        params = {}
    elif data == 'apollo':
        import config.policy_config_apollo as c
        c.mcmc_sample_num = 300
        params = {'stereo_rect': True}

    c.is_crop= args.is_crop
    c.is_rel = args.is_rel
    c.gpu_id = args.gpu_id
    c.prefix = args.prefix
    c.network = args.network
    c.has_network = with_agent
    print [(item, eval('c.' + item)) for item in dir(c) if not item.startswith("__")]

    params = data_libs[data].set_params_disp(**params)
    env = data_libs[data + '_env'].Env(c, params, split='minival')
    sampler = env.sample_state_rect if data == 'apollo' else env.sample_mcmc_state

    models = {'actor': args.init_actor_model}
    agent = pg_car_fit.PGCarFitting(c, env)

    if with_agent:
        agent.pgnet.init_params(models)

    data_transform = data_setting.get_policy_data_setting(env)
    pose_eval_metric = eval_metric.PoseMetric(
             output_names=['pose_output'],
             label_names=env.action_names, is_euler=True,
             trans_idx=[3, 4, 5], rot_idx=[0, 1, 2])

    seg_eval = {'mean_IOU': [], 'mean_delta': [], 'mean_time': []}
    inst = 0
    car_poses = []
    while True:
        state, act = sampler()
        logging.info('%d, %d / %d' % (env.counter, env.inst_counter, env.image_num))
        if env.counter == env.image_num:
            break

        if env.inst_counter == len(env.valid_id):
            uts.mkdir_if_need(params['output_path'] + 'mcmc_%d' % c.mcmc_sample_num)
            file_name = params['output_path'] + 'mcmc_%d/%s.pkl' % (
                    c.mcmc_sample_num, env.image_name)
            pkl.dump(car_poses, open(file_name, 'wb'))
            car_poses = []

        # sample a car instance
        max_reward, max_IOU, max_delta  = [-1 * np.inf for i in range(3)]
        s = time.time()
        if with_agent:
            state_data = OrderedDict({})
            state_data = data_transform_4_network(state,
                          agent.state_names, data_transform)
            # transform state to be input of network
            action = agent.self_action(state_data)
        else:
            action = {'del_pose': np.zeros((1, 6))}

        action, reward, inspect, done, is_better = agent.mcmc_sample(
                   action, state)
        max_reward, max_IOU, max_delta = eval_uts.get_max_reward(
                   reward, inspect, [max_reward, max_IOU, max_delta])
        sample_time = time.time() - s

        idx = np.argmax(reward['reward'])
        car_name = env.car_model.keys()[idx]
        car_poses.append({'pose': action['del_pose'] + state['pose'],
            'car_name': car_name})

        # logging.info("Ep: %d Reward: %f max IOU: %f, max depth rel %f" %
        # (step, max_reward, max_IOU, max_delta))
        pose_eval_metric.update(
                [mx.nd.array(act['del_pose'])],
                [mx.nd.array(action['del_pose'] + state['pose'])])

        # maximum sampled reward
        logging.info("Epoch[%d] Reward=%f max IOU=%f, depth max rel=%f, time=%f" \
                      % (env.counter, max_reward, max_IOU, max_delta, sample_time))
        seg_eval['mean_IOU'].append(max_IOU)
        seg_eval['mean_delta'].append(max_delta)
        seg_eval['mean_time'].append(sample_time)
        inst += 1

    eval_name_vals = pose_eval_metric.get()
    for name, val in zip(eval_name_vals[0], eval_name_vals[1]):
        logging.info("Epoch[%d] Validation-%s=%f" % (env.counter, name, val))

    for name in seg_eval.keys():
        val = np.array(seg_eval[name]).mean()
        per = (np.array(seg_eval[name]) > 0.9).mean()
        logging.info("Epoch[%d] Validation-%s=%f, %f" % (env.counter, name, val, per))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a mx model.')
    parser.add_argument('--init_actor_model', default=None,
        help='The type of fcn-xs model for init, e.g. fcnxs, fcn16s, fcn8s.')
    parser.add_argument('--init_critic_model', default=None,
        help='The type of fcn-xs model for init, e.g. fcnxs, fcn16s, fcn8s.')
    parser.add_argument('--prefix', default='./output/policy-net',
        help='')
    parser.add_argument('--gpu_id', type=int, default=0,
        help='gpu_id to use')
    parser.add_argument('--network', type=str, default='demon',
        help='network use for training')
    parser.add_argument('--epoch', type=int, default=150,
        help='The epoch number of vgg16 model.')
    parser.add_argument('--is_rel', type=uts.str2bool, default='true',
        help='true means we train relative pose.')
    parser.add_argument('--is_crop', type=uts.str2bool, default='true',
        help='true means we train relative pose.')
    parser.add_argument('--dataset', default='kitti',
        help='the data set need to train')
    parser.add_argument('--w_agent', type=uts.str2bool, default='false',
        help='true means we train relative pose.')
    parser.add_argument('--method',  default='pg',
        help='method to train the networks')

    args = parser.parse_args()
    if args.method == 'ddpg':
        train_agent(args.dataset, args.w_agent)
    elif args.method == 'pg':
        train_policy_agent(args.dataset, args.w_agent)
    elif args.method == 'mcmc':
        mcmc_policy_agent(args.dataset, args.w_agent)



