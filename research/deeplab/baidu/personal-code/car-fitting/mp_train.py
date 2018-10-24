from __future__ import division
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
from collections import OrderedDict
import evaluation.eval_utils as eval_uts

import data.kitti as kitti
import data.apolloscape as apollo
import data.kitti_env as kitti_env
import data.apolloscape_env as apollo_env
import utils.utils_3d as uts_3d
import pickle as pkl
import multiprocessing as mp
import subprocess as sp
import mxnet as mx


data_libs = {}
data_libs['kitti'] = kitti
data_libs['apollo'] = apollo
data_libs['kitti_env'] = kitti_env
data_libs['apollo_env'] = apollo_env
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def overwrite_config(c, args):
    c.dataset = args.dataset
    c.nprocs = args.nprocs
    c.gpu_id = args.gpu_id
    c.prefix = args.prefix
    c.network = args.network
    c.is_crop = True
    c.is_rel = args.is_rel # if true the network output is absolute pose
    c.w_agent = args.w_agent


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


def mp_train_policy_agent(data='kitti', with_agent=True):
    """ we do mutiple process mcmc sampling for learning the model
    """
    if data == 'kitti':
        import config.policy_config as c
    elif data == 'apollo':
        import config.policy_config_apollo as c
    else:
        raise ValueError(' no such dataset ')

    # map the args to configuration, initialize the environment
    overwrite_config(c, args)
    print [(item, eval('c.' + item)) for item in dir(c) if not item.startswith("__")]

    params = data_libs[data].set_params_disp()
    env = data_libs[data + '_env'].Env(c, params, split='train')
    sampler_train = env.sample_mcmc_state if data == 'kitti' else env.sample_state_rect

    models = {'actor': args.init_actor_model}
    agent = pg_car_fit.PGCarFitting(c, env)
    agent.pgnet.init_params(models)

    data_transform = data_setting.get_policy_data_setting(env)
    Epoch = 0

    for episode in xrange(c.EPISODES):
        # sample a batch car instance
        state_batch = []
        state_data_batch = []
        act_batch = []
        action_batch = []
        is_eval = False
        for i in range(c.nprocs):
            state, act = sampler_train()
            if (env.counter == env.image_num and \
                    env.inst_counter == len(env.valid_id)):
                is_eval = True

            state_batch.append(state.copy())
            act_batch.append(act)
            state_data = data_setting.data_transform_4_network(
                    state, agent.state_names, data_transform)
            state_data_batch.append(state_data.copy())
            # transform state to be input of network
            action = agent.self_action(state_data)
            action_batch.append(action.copy())

        start = time.time()
        # reset each time
        manager = mp.Manager()
        return_dict = manager.dict()
        procs = []
        for i in range(c.nprocs):
            # change to multi process sampling, initialize many dummy env & agent
            state, action = state_batch[i], action_batch[i]
            key = '%s_%s' % (state['image_name'], state['inst_id'])

            # see whether we can start from already stored exmaple
            act_buf = {}
            reward_buf = {}
            # if agent.replay_buffer.has_state(state):
            #     stat_buf, act_buf, reward_buf = agent.replay_buffer.get_a_sample(key)

            method = mp_mcmc_sampling
            arguments = (c, action, state, act_buf, reward_buf, key, return_dict)
            # method(*arguments)
            proc = mp.Process(target=method, args=arguments)
            proc.daemon = True
            proc.start()
            procs.append(proc)

        # wait all the procs are finished
        for proc in procs:
            proc.join()

        sample_time = time.time() - start
        for i in range(c.nprocs):
            state = state_batch[i]
            state_data = state_data_batch[i]
            key = '%s_%s' % (state['image_name'], state['inst_id'])
            max_reward, max_IOU, max_delta  = [-1 * np.inf for i in range(3)]
            action, reward, inspect, done, is_better = return_dict[key]
            if not is_better:
                continue
            max_reward, max_IOU, max_delta = eval_uts.get_max_reward(
                  reward, inspect, [max_reward, max_IOU, max_delta])
            reward_in = {'reward': max_reward * np.ones((1, 1))}
            agent.perceive(state_data, action, reward=reward_in, done=done)
            learning_time = time.time() - start
            # maximum sampled reward
            # logging.info(action)
            logging.info("Image [%s] Car Inst [%d] Reward: %f, max IOU: %f, depth max rel: %f,  time cost: (sampling: %f learning: %.4f)" \
                    % (state_data['image_name'], state_data['inst_id'], \
                       max_reward, max_IOU, max_delta, \
                       sample_time / c.nprocs, learning_time / c.nprocs))

        # Validation:
        # if episode % env.image_num == 0 and episode > env.image_num:
        if episode == 0 or is_eval:
            agent.save_networks(Epoch)
            exec_eval(agent.get_model_name(Epoch), args)
            Epoch += 1


def exec_eval(model_name, args):
    cmd = 'python evaluation/eval_network.py --gpu=%d --network=%s --model_name=%s --dataset=%s' \
            % (args.eval_gpu_id, args.network, model_name, args.dataset)
    env = os.environ.copy()
    env['DMLC_JOB_CLUSTER'] = 'local'
    sp.call(cmd, shell=True, env=env)


def mp_mcmc_sampling(config, action, state, act_buf, reward_buf, key, return_dict):
    config.w_agent = False
    # p = mp.current_process()
    # print key, p.name, p.pid
    # print [(item, eval('config.' + item)) for item in dir(config) \
    #         if not item.startswith("__")]
    params = data_libs[config.dataset].set_params_disp()

    env = data_libs[config.dataset + '_env'].Env(config, params, split='minival')
    reward_actor, _, _, _ = env.step(action, state)
    if bool(act_buf):
        if np.max(reward_buf['reward']) > np.max(reward_actor['reward']):
            action = act_buf.copy()

    agent = pg_car_fit.PGCarFitting(config, env)
    action, reward, inspect, done, is_better = agent.mcmc_sample(action, state)
    return_dict[key] = [action, reward, inspect, done, is_better]
    # return_dict[key] = action
    # return_dict[key] = p.pid


class MPPolicyMCMCAgent(object):
    def __init__(self, args, split):
        if args.dataset == 'kitti':
            import config.policy_config as c
            in_params = {}

        elif args.dataset == 'apollo':
            import config.policy_config_apollo as c
            c.mcmc_sample_num = 300
            in_params = {'stereo_rect': True}

        c.dataset = args.dataset
        c.is_crop= args.is_crop
        c.is_rel = args.is_rel
        c.gpu_id = args.gpu_id
        c.prefix = args.prefix
        c.network = args.network
        c.w_agent = args.w_agent
        c.nprocs = args.nprocs

        self.config = c
        self.params = data_libs[args.dataset].set_params_disp(**in_params)
        print [(item, eval('c.' + item)) for item in dir(c) if not item.startswith("__")]
        self.num_process = args.nprocs
        self.image_list = [image_name.strip() for image_name in open(
            self.params[split + '_list'])]
        self._procs = []
        self.env = data_libs[args.dataset + '_env'].Env(self.config,
                self.params, split='val')
        self.agent = pg_car_fit.PGCarFitting(self.config, self.env)


    def run(self):
        self._queues = []
        for i in xrange(self.config.nprocs):
            workder = mp_mcmc_policy_agent
            in_args = (self.config, i, self.num_process)
            # res = mp_mcmc_policy_agent(*in_args)
            proc = mp.Process(target=workder, args=in_args,
                    name='sampler-{}'.format(i))
            proc.daemon = True
            proc.start()
            self._procs.append(proc)

        for i in xrange(self.config.nprocs):
            self._procs[i].join()


    def eval(self):
        pose_eval_metric = eval_metric.PoseMetric(
                 output_names=['pose_output'],
                 label_names=['del_pose'], is_euler=True,
                 trans_idx=[3, 4, 5], rot_idx=[0, 1, 2])
        seg_eval = {'mean_IOU': [], 'mean_delta': [], 'mean_time': []}
        for i, image_name in enumerate(self.image_list[:10]):
            if i % 10 == 0:
                logging.info('%d / %d\n' % (i, len(self.image_list)))

            pose_res_file = self.params['output_path'] + 'mcmc_%d/%s.pkl' % (
                    self.config.mcmc_sample_num, image_name)
            carpose = pkl.load(open(pose_res_file,"rb"))
            mask_file = self.data_params['car_inst_path_rect'] + self.image_name + '.png'

            masks = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
            pose_file = self.params['car_pose_path'] + self.image_name + '.poses'
            if not os.path.exists(pose_file) or not os.path.exists(pose_res_file):
                continue

            gt_poses = data_libs[self.d_name].read_carpose(
                    pose_file, is_euler=False)
            valid_id = np.unique(masks)
            # theshold for valid mask
            valid_mask = [(i > 0 and np.sum(self.masks == i) > 10) for i in self.valid_id]
            valid_id = valid_id[valid_mask]

            for res, car_inst_id in zip(carpose, valid_id):
                pose_gt = np.matmul(self.params[self.cam_name + '_ext'],
                            gt_poses[car_inst_id - 1]['pose'])
                rot = uts_3d.rotation_matrix_to_euler_angles(
                       pose_gt[:3, :3], check=False)
                rot[2] += 2 * np.pi
                trans = pose_gt[:3, 3].flatten()
                pose_gt = np.hstack([rot, trans])
                pose_eval_metric.update(
                        [mx.nd.array(pose_gt[None, :])],
                        [mx.nd.array(res['pose'])])

            eval_name_vals = pose_eval_metric.get()
            for name, val in zip(eval_name_vals[0], eval_name_vals[1]):
                logging.info("Validation-%s=%f" % (name, val))

            for name in seg_eval.keys():
                val = np.array(seg_eval[name]).mean()
                per = (np.array(seg_eval[name]) > 0.9).mean()
                logging.info("Validation-%s=%f, %f" % (name, val, per))


def mp_mcmc_policy_agent(config, pid=None, nprocs=None):
    if args.dataset == 'kitti':
        import config.policy_config as c
        params = {}

    elif args.dataset == 'apollo':
        import config.policy_config_apollo as c
        c.mcmc_sample_num = 30
        params = {'stereo_rect': True}

    data = config.dataset
    config.pid = pid
    config.nprocs = nprocs
    params = data_libs[data].set_params_disp(**params)

    env = data_libs[data + '_env'].Env(config, params, split='val')
    sampler = env.sample_state_rect if data == 'apollo' else env.sample_mcmc_state
    agent = pg_car_fit.PGCarFitting(c, env)
    if config.w_agent:
        models = {'actor': args.init_actor_model}
        agent.pgnet.init_params(models)

    data_transform = data_setting.get_policy_data_setting(env)

    inst = 0
    car_poses = []
    while True:
        p = mp.current_process()
        logging.info('%s, %s' % (p.name, p.pid))
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
        if config.w_agent:
            state_data = OrderedDict({})
            state_data = data_setting.data_transform_4_network(state,
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
        # maximum sampled reward
        logging.info("Epoch[%d] Reward=%f max IOU=%f, depth max rel=%f, time=%f" \
                      % (env.counter, max_reward, max_IOU, max_delta, sample_time))
        inst += 1



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
    parser.add_argument('--eval_gpu_id', type=int, default=3,
        help='gpu_id for evaluation')
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
    parser.add_argument('--nprocs', type=int, default=4,
        help='method to train the networks')

    args = parser.parse_args()
    if args.method == 'ddpg':
        train_agent(args.dataset, args.w_agent)

    elif args.method == 'pg':
        mp_train_policy_agent(args.dataset, args.w_agent)

    elif args.method == 'mcmc':
        agent = MPPolicyMCMCAgent(args, 'val')
        agent.run()
        agent.eval()



