"""Test demo for trained model
"""
import argparse
import logging
import mxnet as mx
import numpy as np
import utils.utils as uts
import network.car_pose_net as pose_net
import Networks.net_util as net_util
import evaluation.eval_utils as eval_uts
import algorithm.ddpg_car_fit as car_fit

from collections import namedtuple, OrderedDict
import pdb

import data.kitti as kitti
import data.kitti_env as kitti_env
import data.kitti_iter as kitti_iter

data_libs = {}
data_libs['kitti'] = kitti
data_libs['kitti_env'] = kitti_env
data_libs['kitti_iter'] = kitti_iter

np.set_printoptions(precision=4, suppress=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
Batch = namedtuple('Batch', ['data'])


class PolicyTester(object):
    def __init__(self, env, config, data_params):
        self.config = config
        self.env = env
        params = {}
        params['batch_size'] = config.batch_size
        params['size'] = self.env.image_size
        params['color_map_list'] = data_params['color_map_list']

        self.params = params
        self.net_name = 'actor'
        self.a_name = 'del_pose'


    def my_load_model(self):
        arg_params, _ = net_util.load_mxparams_from_file(
                self.config.test_model + '.params')

        for name, arr in self.actor.arg_dict.items():
            if self.net_name in name:
                self.actor.arg_dict[name][:] = arg_params[name]


    def init(self):
        ctx = mx.gpu(int(self.config.gpu_id))
        inputs = net_util.get_mx_var_by_name(self.env.state_names)
        in_size = np.round(self.env.image_size)
        self.actor_input_shapes = OrderedDict([
            ("image", (1, 3, in_size[0], in_size[1])),
            ("depth", (1, 1, in_size[0], in_size[1])),
            ("mask", (1, 1, in_size[0], in_size[1])),
            ("pose", (1, 6))])

        act_sym = pose_net.pose_block(inputs,
                       self.params, name=self.net_name)
        self.actor = act_sym[self.a_name].simple_bind(
                       ctx=ctx, **self.actor_input_shapes)
        self.my_load_model()

        # model = {'model': self.config.test_model}
        # self.actor = net_util.load_model(model,
        #         data_names=self.actor_input_shapes.keys(),
        #         data_shapes=self.actor_input_shapes.values(),
        #         net=act_sym[self.a_name], ctx=ctx)

    def my_infer(self, state, iter_num=1):
        for name in self.actor_input_shapes.keys():
            if not ('pose' in name):
                self.actor.arg_dict[name][:] = mx.nd.array(
                        self.env.transforms[name](state[name]))
                # self.actor.arg_dict[name][:] = mx.nd.array(
                #         np.zeros(state[name].shape))

        # 4 init and find the best
        max_reward = 0.0
        res = {}
        for rot in self.env.init_rot:
            cur_pose = state['pose'].copy()
            cur_pose[0, 1] = rot
            # self.actor.arg_dict['pose'][:] = mx.nd.array(
            #       self.env.transforms['pose'](cur_pose))

            for name in ['image', 'depth', 'pose', 'mask']:
                print self.actor.arg_dict[name].mean()

            self.actor.forward(is_train=False)
            del_pose = self.actor.outputs[0].asnumpy()
            print del_pose
            reward, _, _, inspector, cur_res = self.env.step(
                    {'del_pose':del_pose}, get_res=True)
            if np.max(reward['reward']) > max_reward:
                max_reward = np.max(reward['reward'])
                res = cur_res


        return res


    def infer(self, state, iter_num=1):
        state_data = OrderedDict({})
        for name in self.actor_input_shapes.keys():
            if not ('pose' in name):
                state_data[name] = mx.nd.array(
                    self.env.transforms[name](state[name]))

        for i in range(iter_num):
            state_data[name] = mx.nd.array(
                self.env.transforms['pose'](self.env.state['pose']))
            self.actor.forward(Batch(state_data.values()))
            outputs = self.actor.get_outputs()
            del_pose = outputs[0].asnumpy()
            self.env.state['pose'] += del_pose
            print self.env.state['pose']
            print del_pose

        reward, _, _, inspector, res = self.env.step(
                {'del_pose':del_pose}, get_res=True)

        uts.plot_images({'mask': res['mask'],
                         'depth': res['depth'],
                         'mask_in': self.env.state['mask']})
        return res



    def save_res(self):
        pass


    def test_policy(self, data_set='kitti'):
        """ test a single network performance
        """
        cur_image_counter = self.env.counter
        total_mask, total_depth, boxes = [np.zeros(self.env.image_size),
               np.zeros(self.env.image_size), []]
        gt_instance = []
        pred_instance = []
        inst_num = 0
        inst_id = 1

        while self.env.counter < len(self.env.image_list):
            state = self.env.sample_state()
            if cur_image_counter == self.env.counter:
                res = self.my_infer(state, iter_num=2)
                self.merge_in(res, inst_id, total_mask, total_depth, boxes)
                inst_num += 1
                inst_id += 1
            else:
                if self.config.vis and self.env.inst_counter > 0:
                    self.vis_all(total_mask, total_depth, boxes)

                if self.config.save:
                    self.save_res(total_mask, total_depth, boxes)

                total_mask, total_depth, boxes = [np.zeros(self.env.image_size), np.zeros(self.env.image_size), []]
                inst_id = 1
                cur_image_counter = self.env.counter

                res = self.my_infer(state, iter_num=2)
                self.merge_in(res, inst_id, total_mask, total_depth, boxes)
                inst_num += 1
                inst_id += 1

        eval_uts.eval_instance_depth(gt_instance, pred_instance, inst_num)


def test(dataset='kitti'):
    import config.policy_config as config
    config.test_model = args.test_model
    config.gpu_id = args.gpu_id
    config.vis = args.vis
    config.save = args.save

    params = data_libs[dataset].set_params_disp()
    env = data_libs[dataset + '_env'].Env(config, params)
    tester = PolicyTester(env, config, params)
    tester.init()
    tester.test_policy()


class DDPGPolicyTester(object):
    def __init__(self, config, env, agent, params):
        self.env = env
        self.agent = agent
        self.params = params
        self.config = config


    def infer(self, state):
        state_data = state.copy()
        for name in state.keys():
            if not ('pose' in name):
                state_data[name] = self.env.transforms[name]['transform'](
                        state[name])

        max_reward, max_IOU, max_delta  = [0.0 for i in range(3)]
        max_res = {}

        for j in xrange(self.env.timestep_limit):
            action = self.agent.self_action(state_data)
            print action
            reward, done, next_state, inspect, res = self.env.step(action, get_res=True)
            print np.max(inspect['IOU'])
            uts.plot_images({'mask': res['mask'],
                             'depth': res['depth'],
                             'mask_in': self.env.state['mask']})

            if self.config.update_pose:
                state.update(next_state)
                state_data.update(next_state)

            if np.max(reward['reward']) > max_reward:
                max_reward = np.max(reward['reward'])
                idx = np.argmax(reward['reward'])
                max_IOU = max(max_IOU, inspect['IOU'][idx])
                max_delta = max(max_delta, inspect['delta'][idx])
                max_res = res

        return max_res



    def test_policy(self):
        cur_image_counter = self.env.counter
        total_mask, total_depth, boxes = [np.zeros(self.env.image_size),
               np.zeros(self.env.image_size), []]
        gt_instance = []
        pred_instance = []
        inst_num = 0
        inst_id = 1
        while self.env.counter < len(self.env.image_list):
            state = self.env.sample_state()
            if cur_image_counter == self.env.counter:
                res = self.infer(state)
                self.merge_in(res, inst_id, total_mask, total_depth, boxes)
                inst_num += 1
                inst_id += 1
            else:
                print self.env.counter
                if self.config.vis and self.env.inst_counter > 0:
                    self.vis_all(total_mask, total_depth, boxes)

                if self.config.save:
                    self.save_res(total_mask, total_depth, boxes)

                total_mask, total_depth, boxes = [np.zeros(self.env.image_size), np.zeros(self.env.image_size), []]
                inst_id = 1
                cur_image_counter = self.env.counter

                res = self.infer(state)
                self.merge_in(res, inst_id, total_mask, total_depth, boxes)
                inst_num += 1
                inst_id += 1

        eval_uts.eval_instance_depth(gt_instance, pred_instance, inst_num)



def test_ddpg(dataset='kitti'):
    import config.config as config
    config.test_model = args.test_model
    config.gpu_id = args.gpu_id
    config.vis = args.vis
    config.save = args.save

    params = data_libs[dataset].set_params_disp()
    env = data_libs[dataset + '_env'].Env(config, params)
    agent = car_fit.DDPGCarFitting(env, config)
    agent.load_networks({'actor': args.test_model})
    tester = DDPGPolicyTester(config, env, agent, params)
    tester.test_policy()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test the trained model for acc.')
    parser.add_argument('--test_model', default='./output/policy-net-4init-actor-0212', help='The model name with mxnet format.')
    parser.add_argument('--gpu_id', default="3",
        help='the gpu ids use for training')
    parser.add_argument('--vis', action="store_true", default=True,
        help='whether to save the results')
    parser.add_argument('--save', action="store_true", default=False,
        help='whether to save the results')

    args = parser.parse_args()
    logging.info(args)

    # test('kitti')
    test_ddpg()




