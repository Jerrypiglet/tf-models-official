from __future__ import division
import os
import argparse
import numpy as np
import algorithm.mcmc_car_fit as car_fit
import logging
import config.config as c
import evaluation.eval_utils as eval_uts
import utils.utils as uts
import pdb
import pickle as pkl

import data.kitti as kitti
import data.kitti_env as kitti_env
import data.kitti_iter as kitti_iter

data_libs = {}
data_libs['kitti'] = kitti
data_libs['kitti_env'] = kitti_env
data_libs['kitti_iter'] = kitti_iter

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
# print os.environ['MXNET_ENGINE_TYPE']

class MCMCPolicyTester(object):
    def __init__(self, config, env, agent, params):
        self.env = env
        self.agent = agent
        self.params = params
        self.config = config
        self.max_depth = 1000.0


    def test(self):
        total_mask, total_depth, boxes = [
                        np.zeros(self.env.image_size, dtype=np.uint32), \
                        self.max_depth * np.ones(self.env.image_size), []]
        res_all = []
        gt_instance = []
        pred_instance = []
        inst_num = 0
        inst_id = 1

        while self.env.counter < len(self.env.image_list):
            state = self.env.sample_state()
            # initialize the fitting
            action = self.agent.fit(state, verbose=0)
            _, _, _, ins, res = self.env.step(
                      action, state, get_res=True)
            res_all.append(res)
            eval_uts.merge_inst(
                    res, inst_id, total_mask, total_depth, boxes)
            inst_num += 1
            inst_id += 1

            if self.env.inst_counter == len(self.env.masks):
                if self.config.vis and inst_num >= 1:
                    state = self.env.state.copy()
                    state['masks'] = self.env.masks
                    eval_uts.vis_res(total_mask, total_depth, boxes,
                            state, self.params['color_map_list'])

                if self.config.save and inst_num >= 1:
                    self.save_res(res_all,
                            total_mask, total_depth, boxes, \
                            self.env.image_name)

                # reset
                logging.info('image %d' % (self.env.counter - 1))
                total_mask, total_depth, boxes = [\
                        np.zeros(self.env.image_size, dtype=np.uint32), \
                        self.max_depth * np.ones(self.env.image_size), []]
                res_all = []
                inst_id = 1

        eval_uts.eval_instance_depth(gt_instance, pred_instance, inst_num)


    def save_res(self, res_all, total_mask, total_depth, boxes, save_name):
        pkl.dump(res_all, open(self.params['save_path'] + save_name + '.pkl', 'wb'))
        pkl.dump(total_mask, open(self.params['save_path'] + save_name + '_mask.pkl', 'wb'))
        pkl.dump(total_depth, open(self.params['save_path'] + save_name + '_depth.pkl', 'wb'))
        logging.info('saved %s' % save_name)


def mcmc_test(dataset='kitti'):
    import config.config as config
    config.vis = args.vis
    config.save = args.save

    params = data_libs[dataset].set_params_disp()
    params['save_path'] = params['output_path'] + 'mcmc/'
    uts.mkdir_if_need(params['save_path'])

    env = data_libs[dataset + '_env'].Env(config, params)
    agent = car_fit.MCMCCarFitting(config, env, params)
    tester = MCMCPolicyTester(config, env, agent, params)
    tester.test()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a mx model.')
    parser.add_argument('--save_path', default='../output/',
        help='')
    parser.add_argument('--vis', action='store_true', default=False,
        help='true means visualization.')
    parser.add_argument('--save', action='store_true', default=True,
        help='true means save results.')
    args = parser.parse_args()
    logging.info(c)
    mcmc_test()

