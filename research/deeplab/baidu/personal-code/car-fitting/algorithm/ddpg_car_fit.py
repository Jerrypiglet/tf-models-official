from __future__ import division
import os
import cv2
import pdb
import numpy as np
import mxnet as mx
import data.replay_buffer as rb
import network.car_pose_net as net
import ou_noise as noise
from collections import OrderedDict
import logging


def is_exists(fields, name):
    for field in fields:
        if field in name:
            return True

    return False


class DDPGCarFitting(object):
    """docstring for DDPG"""

    def __init__(self, env, config):
        mx.random.seed(config.seed)
        np.random.seed(config.seed)

        self.env = env
        self.action_names = ['del_pose']
        self.transforms = OrderedDict({'del_pose': env.transforms['pose']})
        assert len(self.action_names) == 1
        self.action_range = [[0.0, 0.0],
                             [0, 2 * np.pi],
                             [0.0, 0.0]] + \
                             [[-3.0, 3.0],
                              [-1, 1],
                              [-3.0, 3.0]]
        self.action_dim = 6
        self.action = OrderedDict({self.action_names[0]: \
                np.zeros(self.action_dim)})

        if config.gpu_flag:
            self.ctx = mx.gpu(config.gpu_id)
        else:
            self.ctx = mx.cpu()

        self.ddpgnet = net.DDPGNet(config, env, self)
        self.exploration_noise = noise.OUNoise(self.action_dim)
        self.replay_buffer = rb.ReplayBuffer(config.memory_size,
                self.ddpgnet.actor_in_names,
                env.action_names,
                env.state_update_names)

        self.batch_size = config.batch_size
        self.config = config

        self.ddpgnet.init()
        self.train_step = 0


    def train(self):
        # Sample a random minibatch of N transitions from replay buffer
        obs_batch, action_batch, reward_batch, next_batch, done_batch = \
                self.replay_buffer.get_batch(self.batch_size)
        next_state_batch = obs_batch.copy()
        next_state_batch.update(next_batch)

        # print 'obs pose %s' % (obs_batch['pose'])
        # print 'next pose %s' % (next_batch['pose'])

        # Calculate y_batch
        next_qvals = self.ddpgnet.get_target_q(next_state_batch).asnumpy()
        y_batch = []
        for i in range(self.batch_size):
            if done_batch[i]:
                y_batch.append(reward_batch['reward'][i, :])
            else:
                y_batch.append(reward_batch['reward'][i, :] + \
                        self.config.GAMMA * next_qvals[i, :])

        y_batch = np.resize(y_batch,
                [self.batch_size, self.env.data_params['car_num']])

        # Update critic by minimizing the loss L
        self.ddpgnet.update_critic(obs_batch, action_batch,
                {'reward':y_batch})

        # Update actor by maxmizing Q
        self.ddpgnet.update_actor(obs_batch)
        self.train_step += 1

        # update target networks
        self.ddpgnet.update_target()


    def random_action(self):

        act = noise.uniform_sample(self.action_range)
        self.action[self.action_names[0]] = act
        return self.action


    def noise_action(self, state, action=None):
        #Select action a_t according to the current policy and exploration noise
        if action is None:
            action = self.ddpgnet.get_step_action(state)
        else:
            action = action[self.action_names[0]]

        self.action[self.action_names[0]] = \
                action + self.exploration_noise.noise()

        return self.action


    def self_action(self, state):
        self.action[self.action_names[0]] = \
            self.ddpgnet.get_step_action(state)
        return self.action


    def perceive(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > self.config.memory_start_size:
            self.train()

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()

    def save_networks(self, episode):
        self.ddpgnet.save_networks('actor', self.config.prefix,
                episode)
        self.ddpgnet.save_networks('critic', self.config.prefix,
                episode)

    def load_networks(self, model_name):
        logging.info('loading initial networks')
        self.ddpgnet.load_networks('actor', model_name['actor'])
        self.ddpgnet.load_networks('critic', model_name['critic'])



