from __future__ import division
import os
from collections import OrderedDict
import logging

import pdb
import numpy as np
import ou_noise as noise
import utils.utils as uts
import matplotlib.pyplot as plt


class MCMCCarFitting(object):
    def __init__(self, config, env, params):
        self.num_samples = 500
        self.config = config
        self.params = params
        self.env = env

        self.action_names = ['del_pose']
        self.transforms = OrderedDict({'del_pose': env.transforms['pose']})
        assert len(self.action_names) == 1
        self.action_range = [[0.0, 0.0],
                             [0, 2 * np.pi],
                             [0.0, 0.0]] + \
                            [[-3.0, 3.0],
                             [-0.5, 0.5],
                             [-3.0, 3.0]]
        self.action_noise_sigma = np.float32([0.0, 0.1, 0.0, 0.1, 0.01, 0.1])
        self.action_dim = 6
        self.action = OrderedDict({self.action_names[0]: \
                np.zeros((1, self.action_dim))})
        self.exploration_noise = noise.OUNoise(self.action_dim)


    def random_sample(self, action, state, rot, sample_ori=True):
        idx = np.random.randint(1)
        logq = 0.0

        if idx == 0:
            mu = self.env.init_pose(is_rand=True)
            mu = mu[:, 3:]
            sigma = 0.5 * np.eye(3)
            reset_pose = mu + np.random.randn(1, 3) * np.diag(sigma)

            action = np.zeros((1, self.action_dim))
            action[:, 3:] =  reset_pose - state['pose'][:, 3:]
            mu = np.transpose(mu)
            logq = noise.mvnlogpdf(reset_pose, mu, sigma) - \
                    noise.mvnlogpdf(state['pose'][:, 3:], mu, sigma)

        else:
            act = np.random.randint(2, size=self.action_dim)
            action = action + self.exploration_noise.noise_multi(
                    sigma=self.action_noise_sigma, num=1, act=act)

        if not sample_ori:
            action[:, 2] = rot

        # hard limit for action range
        action[:, 2] = min(max(
            action[:, 2], -1 * np.pi / 4.), np.pi / 4.)

        # for i, r in enumerate(self.action_range):
        r = self.action_range[4]
        action[:, 4] = min(max(action[:, 4], r[0]), r[1])

        return action, logq


    def fit(self, state, verbose=1):
        """ Fit using max reward
        """
        cur_action = self.action[self.action_names[0]].copy()
        reward, _, _, _ = self.env.step(self.action, state)
        cur_reward = np.max(reward['reward'])
        self.reward = cur_reward

        if verbose == 2:
            plt.ion()
            fig = plt.figure(figsize=(10, 5))

        for rot in self.env.init_rot:
            logging.info('rot %s' % rot)
            cur_action[:, 2] = rot

            for i in range(self.num_samples):
                sample_ori = i > self.num_samples / 10

                # sampling based cur action state and given rotation
                action, logq = self.random_sample(
                        cur_action, state, rot, sample_ori)
                reward, _, _, _ = self.env.step(
                        {self.action_names[0]: action},
                         state)

                Temp = 200.0 / (200.0 + i)
                delta = np.log(np.max(reward['reward'])) - \
                        np.log(cur_reward)

                ac_rate = min(1.0, np.exp((logq + delta) / Temp))
                ac = np.random.rand() < ac_rate
                if ac:
                    cur_reward = np.max(reward['reward'])
                    cur_action = action

                if cur_reward > self.reward:
                    self.action[self.action_names[0]] = cur_action
                    self.reward = cur_reward

                if i % (self.num_samples / 10) == 0 and verbose >= 1:
                    reward, _, _, ins, res = self.env.step(
                        self.action, state, get_res=True)
                    idx = np.argmax(reward['reward'])
                    logging.info('cur_action: %s' % cur_action)
                    logging.info('eval reward %s iou %s, delta %s' % (
                        self.reward,
                        ins['IOU'][idx],
                        ins['delta'][idx]))

                    if verbose == 2:
                        res.update({'ori_mask': state['mask'], \
                                    'ori_depth': state['depth']})
                        uts.plot_images(res, fig=fig)

        return self.action


    def fit_single(self, state):
        """ This means we fit sample num for each model
        """
        pass




