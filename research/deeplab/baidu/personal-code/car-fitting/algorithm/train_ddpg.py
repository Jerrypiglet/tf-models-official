from __future__ import division
import os
import sys
import time
import math
import nets
import pickle
import random
import numpy as np
import mxnet as mx
import scipy.misc as sm
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils as uts
import config as c
import replay_buffer as rb
import ou_noise as noise
import network as net


def is_exists(fields, name):
    for field in fields:
        if field in name:
            return True

    return False


class DDPGCarFitting(object):
    """docstring for DDPG"""

    def __init__(self, c):
        mx.random.seed(c.seed)
        np.random.seed(c.seed)
        self.ctx = mx.gpu([int(i) for i in c.gpu_ids.split(',')])

        self.ddpgnet = net.DDPGNet(c)
        self.exploration_noise = noise.OUNoise(self.action_dim)
        self.replay_buffer = rb.ReplayBuffer(c.memory_size)

        self.batch_size = c.batch_size

        self.ddpgnet.init()
        self.train_step = 0


    def train(self):
        # print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(self.batch_size)

        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch, [self.batch_size, self.action_dim])

        # Calculate y_batch
        next_qvals = self.ddpgnet.get_target_q(next_state_batch).asnumpy()

        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + c.GAMMA * next_qvals[i][0])
        y_batch = np.resize(y_batch, [self.batch_size, 1])

        # Update critic by minimizing the loss L
        self.ddpgnet.update_critic(state_batch, action_batch, y_batch)

        # Update actor by maxmizing Q
        self.ddpgnet.update_actor(state_batch)
        self.train_step += 1

        # update target networks
        self.ddpgnet.update_target()


    def noise_action(self, state):
        # Select action a_t according to the current policy and exploration noise
        state = np.reshape(state, (1, self.state_dim))
        action = self.ddpgnet.get_step_action(state)
        return action + self.exploration_noise.noise()


    def action(self, state):
        state = np.reshape(state, (1, self.state_dim))
        action = self.ddpgnet.get_step_action(state)
        return action


    def perceive(self, state, action, reward, next_state, done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > c.memory_start_size:
            self.train()

            # if self.time_step % 10000 == 0:
            # self.actor_network.save_network(self.time_step)
            # self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()


