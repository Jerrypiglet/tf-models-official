from __future__ import division
import os
import cv2
import pdb
import argparse
import numpy as np
import mxnet as mx
import logging
import config.policy_config as c
import data.replay_buffer as rb
import train_policy as policy
import ou_noise as noise
from collections import OrderedDict
import utils.utils as uts
import data.kitti as kitti
import data.kitti_env as kitti_env
import data.kitti_iter as kitti_iter

data_libs = {}
data_libs['kitti'] = kitti
data_libs['kitti_env'] = kitti_env
data_libs['kitti_iter'] = kitti_iter


def is_exists(fields, name):
    for field in fields:
        if field in name:
            return True

    return False


class PGCarFitting(object):
    """docstring for DDPG"""

    def __init__(self, config, env):
        np.random.seed(config.seed)

        self.num_samples = config.mcmc_sample_num
        self.env = env
        self.action_names = ['del_pose']
        assert len(self.action_names) == 1
        self.action_range = [[0.0, 0.0],
                             [0, 2 * np.pi],
                             [0.0, 0.0]] + \
                            [[-5.0, 5.0],
                             [-3.0, 3.0], # height
                             [-5.0, 5.0]]
        self.action_noise_sigma = np.array([0.01, 0.3, 0.01, 0.1, 0.1, 0.1])
        self.action_dim = len(self.action_range)
        self.action = OrderedDict({self.action_names[0]: \
                np.zeros(self.action_dim)})

        self.reward_name = config.reward_name
        self.state_names = ['image', 'depth', 'mask', 'render_depth', 'pose']
        if config.is_crop:
            self.state_names = ['image', 'depth', 'mask', 'crop', 'pose']
        self.exploration_noise = noise.OUNoise(self.action_dim)

        # self.replay_buffer = rb.ReplayBuffer(
        #       config.memory_size,
        #       self.state_names, self.action_names)
        self.replay_buffer = rb.InstanceSetBuffer(
                        self.state_names, self.action_names,
                        config.memory_size)
        self.net_name = 'actor'
        self.batch_size = config.batch_size
        self.config = config

        if config.w_agent:
            mx.random.seed(config.seed)
            self.ctx = mx.gpu(config.gpu_id)
            self.pgnet = policy.PG_trainer(
                config, env, state_names=self.state_names)
            self.pgnet.init()

        self.mp = 'nprocs' in dir(config)
        self.train_step = 0


    def train(self):
        # Sample a random minibatch of N transitions from replay buffer
        obs_batch, action_batch, reward_batch, next_batch, done_batch = \
                self.replay_buffer.get_batch(self.batch_size)
        next_state_batch = obs_batch.copy()
        next_state_batch.update(next_batch)

        # Update actor by maxmizing Q
        self.pgnet.update_actor(obs_batch, action_batch, reward_batch)
        self.train_step += 1


    def random_action(self):
        # do a random action
        act = noise.uniform_sample(self.action_range)
        self.action[self.action_names[0]] = act
        return self.action


    def step_decay(self, step):
        """ drop sigma by half every 10 times fitting
        """
        drop = 0.5
        step_drop = self.env.image_num * 10.0
        lrate = np.power(drop, np.floor((1+step)/step_drop))
        return lrate


    def random_sample(self, action, state, rot):
        idx = np.random.randint(2)
        idx = 1
        logq = 0.0
        dim = action.shape[1]
        if idx == 0:
            mu = self.env.init_pose(is_rand=True,
                    state=state if self.mp else None)
            mu = mu[:, 3:]
            sigma = 0.5 * np.eye(3)
            reset_pose = mu + np.random.randn(1, 3) * np.diag(sigma)

            action = np.zeros((1, dim))
            action[:, 3:] = reset_pose - state['pose'][:, 3:]
            action[:, 1] = rot - state['pose'][:, 1]

            # set orientation has one direction
            mu = np.transpose(mu)
            logq = noise.mvnlogpdf(reset_pose, mu, sigma) - \
                     noise.mvnlogpdf(state['pose'][:, 3:], mu, sigma)

        else:
            decay = self.step_decay(self.train_step)
            sigma = self.action_noise_sigma * decay
            activate = np.random.randint(2, size=self.action_dim)
            action = action + self.exploration_noise.noise_multi(
                    sigma=sigma, num=1, activate=activate)

        # hard limit for action range
        # for idx in range(3):
        #     action[:, idx] = min(max(action[:, idx], -1 * state['pose'][:, idx]),
        #                 2 * np.pi - state['pose'][:, idx])

        # for i, r in enumerate(self.action_range):
        # r = self.action_range[4]
        # action[:, 4] = min(max(action[:, 4], r[0]), r[1])

        return action, logq


    def mcmc_sample(self, in_action, state):
        a_name = self.action_names[0]
        action_out = in_action[a_name]
        cur_action = action_out.copy()

        # print state['pose'], state['crop']
        reward_all, done, _, ins = self.env.step(
                {a_name: cur_action}, state)

        reward_out = np.max(reward_all['reward'])
        # key = '%s_%s' % (state['image_name'], state['inst_id'])
        # logging.info(key)
        # logging.info(cur_action)
        # logging.info(reward_out)

        cur_reward = max(reward_out, 1e-10)
        max_invalid_steps = self.num_samples / 10
        is_done = False
        is_better = False
        for rot in self.env.init_rot:
            invalid_step = 0
            for i in range(self.num_samples):
                # sampling based cur action state and given rotation
                action, logq = self.random_sample(cur_action, state, rot)
                reward, done, _, inspector = self.env.step({a_name: action}, state)
                        # is_plot=True)

                reward_max = max(np.max(reward['reward']), 1e-10)
                Temp = 100.0 / (100.0 + i)
                # convert reward to err
                delta = (np.log(reward_max) - np.log(cur_reward)) * 10.0
                ac_rate = min(1.0, np.exp((logq + delta) / Temp))
                ac = np.random.rand() < ac_rate
                if ac:
                    cur_reward = reward_max
                    cur_action = action

                # print ac, ac_rate, cur_action
                if cur_reward > reward_out:
                    action_out = cur_action
                    reward_out = cur_reward
                    reward_all = reward
                    ins = inspector
                    invalid_step = 0
                    is_better = True
                    # logging.info('update: %f' % (cur_reward))

                else:
                    if invalid_step >= max_invalid_steps:
                        break
                    invalid_step += 1

                if done:
                    is_done = True
                    break;

            if is_done:
                break;

        # check whether the sampled examples are put correct
        # print action_out, state['pose']
        # uts.plot_images({'mask': res_all['mask'],
        #                  'depth': res_all['depth'],
        #                  'mask_in': state['mask'],
        #                  'depth_in': state['depth']})
        return {a_name: action_out}, reward_all, ins, done, is_better


    def noise_action(self, state):
        """ init an action, either start from current action,
            or globally restart an action
        """
        cur_action = self.pgnet.get_step_action(state)
        cur_action = \
               cur_action + self.exploration_noise.noise()

        return {self.action_names[0]: cur_action}


    def self_action(self, state, w_loss=False):
        """ w_loss mesure the distance between action and the sampled
            action in replay buffer
        """
        if w_loss:
            self.action[self.action_names[0]], loss = \
                self.pgnet.get_step_action(state, w_loss)
            return self.action, loss
        else:
            self.action[self.action_names[0]] = \
                self.pgnet.get_step_action(state, w_loss)
            return self.action


    def perceive(self, state, action,
            reward, next_state=None, done=False):
        next_state = {} if next_state is None else next_state
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > self.config.memory_start_size:
            self.train()

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()


    def save_networks(self, episode):
        self.pgnet.save_network(episode)


    def get_model_name(self, episode):
        return '%s-%s-%04d' % (self.config.prefix, \
                self.pgnet.net_name, episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a mx model.')
    parser.add_argument('--init_model', default=None,
        help='initial model')
    parser.add_argument('--prefix', default='../Output/',
        help='')
    parser.add_argument('--test_step', type=int, default=10,
        help='Step for optimization')
    parser.add_argument('--epoch', type=int, default=150,
        help='The epoch number of vgg16 model.')
    parser.add_argument('--retrain', action='store_true', default=False,
        help='true means continue training.')

    args = parser.parse_args()
    logging.info(c)
