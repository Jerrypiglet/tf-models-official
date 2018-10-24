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


class metric(object):
    def __init__(self, keys):
        self.values = OrderedDict(
                zip(keys, [0.0 for i in range(len(keys))]))
        self.count = 0

    def update(self, values):
        for name, value in values.items():
            self.values[name] += value
        self.count += 1.0

    def get(self):
        vals = self.values.copy()
        for key, val in vals.items():
            vals[key] /= self.count
        return vals

    def reset(self):
        keys = self.values.keys()
        self.values = dict(zip(keys, [0.0 for i in range(len(keys))]))
        self.count = 0


class PG_trainer(object):
    def __init__(self, c, env,
            data_iter=None,
            state_names=None,
            state_shapes=None):

        self.env = env
        self.c = c
        self.network = c.network
        self.data_iter = data_iter

        self.state_names = env.state_names if state_names is None \
                else state_names
        self.action_names = env.action_names
        self.action_dim = env.action_dim
        in_size = [c.height, c.width] if self.c.is_crop else \
                env.image_size

        self.state_shapes = OrderedDict({
            "image": (self.c.batch_size, 3, in_size[0], in_size[1]),
            "depth": (self.c.batch_size, 1, in_size[0], in_size[1]),
            "render_depth": (self.c.batch_size, 1, in_size[0], in_size[1]),
            "mask": (self.c.batch_size, 1, in_size[0], in_size[1]),
            "pose": (self.c.batch_size, self.action_dim),
            "crop": (self.c.batch_size, 4),
            "reward": (self.c.batch_size, 1),
            "del_pose": (self.c.batch_size, self.action_dim)})

        self.one_state_shape = dict([(key,
            tuple([1] + list(shape[1:]))) for \
            key, shape in self.state_shapes.items()])

        self.state = net_util.get_mx_var_by_name(self.state_names)
        self.action = net_util.get_mx_var_by_name(self.action_names)
        self.reward = net_util.get_mx_var_by_name(env.reward_names)

        self.sampler = noise.OUNoise(self.action_dim)
        names = env.reward_names + env.inspector_names
        self.inspectors = OrderedDict(zip(names, np.zeros(len(names))))
        self.log_step = 5
        self.lr = 0.1

        if c.gpu_flag:
            self.ctx = mx.gpu(c.gpu_id)
        else:
            self.ctx = mx.cpu()


    def init(self, init_model=None):
        self.a_name = self.action_names[0]
        self.net_name = 'pose'

        params = {}
        params['batch_size'] = self.c.batch_size
        params['size'] = self.env.image_size
        params['crop_size'] = [self.c.height, self.c.width]

        # use the geometric projection loss
        if self.network == 'demon':
            if self.c.is_crop:
                act_sym = pose_net.pose_block_w_crop(self.state,
                  params, name=self.net_name, is_rel=self.c.is_rel)
            else:
                act_sym = pose_net.pose_block(self.state,
                      params, name=self.net_name, is_rel=self.c.is_rel)

        elif self.network == 'resnet':
            act_sym = pose_net.resnet_pose_block(self.state)

        if self.data_iter is None:
            loss = losses.my_pose_loss(act_sym[self.a_name],
                                       self.action[self.a_name],
                                       batch_size=self.c.batch_size,
                                       weight=self.reward['reward'])
            outputs = [mx.sym.BlockGrad(
                    act_sym[self.a_name], name='pose'), loss]
            act_out = mx.symbol.Group(outputs)
        else:
            act_out = act_sym[self.a_name]

        logging.info('Init network symbol')
        if not (self.data_iter is None):
            self.in_shape = OrderedDict(self.data_iter.provide_data)
        else:
            in_net_names = self.state_names + self.action_names + \
                    self.env.reward_names
            self.in_shape = OrderedDict(zip(in_net_names, \
                    [self.state_shapes[name] for name in in_net_names]))
            self.one_in_shape = OrderedDict(zip(in_net_names, \
                    [self.one_state_shape[name] for name in in_net_names]))

        self.actor = act_out.simple_bind(ctx=self.ctx, **self.in_shape)
        self._actor_sym = act_sym[self.a_name]

        self.actor_one = self.actor.reshape(**self.one_in_shape)
        self.actor_updater = mx.optimizer.get_updater(mx.optimizer.create(
            self.c.actor_updater, learning_rate=self.c.actor_lr))
        # pdb.set_trace()

        self.epoch = 0
        self.metric = metric(['reward', 'IOU', 'delta'])


    def init_params(self, init_model=None, allow_missing=True):
        self.actor_state = {}
        for name, arr in self.actor.arg_dict.items():
            if self.net_name in name and not (name in self.state_names):
                initializer = mx.initializer.Uniform(self.c.init_scale)
                initializer._init_weight(name, self.actor.arg_dict[name])
                shape = self.actor.arg_dict[name].shape
                self.actor_state[name] = (mx.nd.zeros(shape, self.ctx),
                        mx.nd.zeros(shape, self.ctx))

        if not (init_model is None):
            logging.info('loading model %s' % init_model)
            symbol, arg_params, aux_params = \
                    net_util.args_from_models(init_model,
                            with_symbol=False)

            for name, arr in self.actor.arg_dict.items():
                if self.net_name in name and not (name in self.state_shapes.keys()):
                    if not (name in arg_params.keys()):
                        if not allow_missing:
                            raise ValueError('%s is missing' % name)
                        else:
                            continue
                    self.actor.arg_dict[name][:] = arg_params[name]


    def get_step_action(self, state, w_loss=False):
        # single observation
        for name in self.state_names:
            self.actor_one.arg_dict[name][:] = state[name]

        self.actor_one.forward(is_train=False)
        if w_loss:
            return self.actor_one.outputs[0].asnumpy(), \
                    self.actor_one.output[1].asnumpy()
        else:
            return self.actor_one.outputs[0].asnumpy()


    def train(self):
        epoch = 0
        while epoch < self.c.epoch:
            data_iter = iter(self.data_iter)
            end_of_batch = False
            next_batch = next(data_iter)
            nbatch = 0
            while not end_of_batch:
                data_batch = next_batch
                self.update(data_batch)
                self.eval_batch()
                if nbatch % self.log_step == 0:
                    self.log_batch(epoch, nbatch)

                try:
                    next_batch = next(data_iter)
                    self.cur_crops = data_iter.get_crops()
                except:
                    end_of_batch = True

                nbatch += 1

            self.metric.reset()
            self.save_network(epoch)
            self.data_iter.reset()
            epoch += 1


    def save_network(self, epoch):
        prefix = self.c.prefix
        param_name = '%s-%s-%04d.params' % (prefix, self.net_name, epoch)
        self._actor_sym.save('%s-%s-symbol.json' % (prefix, self.net_name))
        save_dict = { k : v.as_in_context(mx.cpu()) \
                      for k, v in self.actor.arg_dict.items()}
        mx.ndarray.save(param_name, save_dict)
        logging.info('Save checkpoint to %s' % param_name)


    def eval_batch(self):
        state_names = self.cur_state.keys()
        action = self.actor.outputs[0].asnumpy()
        for key in self.inspectors.keys():
            self.inspectors[key] = 0.0

        for i, cur_act in enumerate(action):
            cur_state = OrderedDict({})
            for name in state_names:
                cur_state[name] = self.cur_state[name][i]
            reward, _, _, inspector = self.env.step(
                 {'del_pose': cur_act[None, :]}, cur_state)
            inspector.update(reward)
            for key, value in inspector.items():
                self.inspectors[key] += np.max(value)

        for key, val in self.inspectors.items():
            self.inspectors[key] /= self.c.batch_size

        self.metric.update(self.inspectors)


    def log_batch(self, epoch, nbatch):

        res = self.metric.get()
        reward_str = ''
        for key, val in res.items():
            reward_str += '%s: %f ' % (key, val)
        logging.info('Epoch[%d], Batch [%d], %s' % (epoch, nbatch, reward_str))


    def get_cur_grad(self, act, noises, cur_state):
        """ Using one step policy gradient to calculate gradient
        """
        reward, _, _, ins = self.env.step(
                            {'del_pose': act[None, :]},
                             cur_state)
        self_reward = np.max(reward['reward'])

        total_reward = 0.0
        grad = np.zeros(act.shape)
        for noi in noises:
            cur_action = act - noi
            reward, _, _, _ = self.env.step(
                    {'del_pose':cur_action[None, :]},
                    cur_state)
            max_reward = np.max(reward['reward'])
            # change to average reward
            if max_reward > self_reward:
                cur_reward = max_reward - self_reward
                grad += noi * cur_reward
                total_reward += cur_reward

        if total_reward > 0:
            grad /= total_reward

        return grad


    def get_gradient(self):
        # get current state
        self.cur_state = OrderedDict({})
        cur_state = OrderedDict({})
        state_names = self.state_names[:]

        # revert transform
        for name in state_names:
            if 'image' in name:
                continue
            tmp = self.actor.arg_dict[name].asnumpy().copy()
            self.cur_state[name] = self.env.transforms[name]['transform'](
                    tmp, forward=False)
            if self.crop_image:
                if self.env.transforms[name]['is_image']:
                    self.cur_state[name] = np.zeros([self.c.batch_size] + \
                            self.env.image_size)
                else:
                    self.cur_state[name] = self.cur_state[name]


        if not ('pose' in self.state_names):
            state_names += ['pose']
            self.cur_state['pose'] = np.zeros((self.c.batch_size, \
                    self.action_dim))

        # revert crop
        if self.crop_image:
            is_pad = np.zeros(self.c.batch_size, dtype=np.bool)
            for i in range(self.c.batch_size):
                crop = self.cur_crops[i]
                if np.sum(crop) == 0:
                    is_pad[i] = True
                    continue

                sz = np.uint32([crop[2] - crop[0], crop[3] - crop[1]])
                for name in state_names:
                    if self.env.transforms[name]['is_image']:
                        inter = self.env.transforms[name]['interpolation']
                        tmp = cv2.resize(cur_state[name][i],
                                tuple(sz[::-1]),
                                interpolation=inter)
                        self.cur_state[name][i] = uts.crop_image(
                                tmp, crop,
                                forward=False,
                                image_size=self.env.image_size)

        action = self.actor.outputs[0].asnumpy()
        sigma = np.array([0., 0.2, 0., 0.1, 0.1, 0.1]) * self.lr
        noises = self.sampler.noise_multi(
                        sigma=sigma,
                        num=self.c.grad_dir_num)
        max_grad = np.zeros(action.shape)

        for i, act in enumerate(action):
            if self.crop_image:
                if is_pad[i]:
                    continue
                if np.sum(sz) == 0:
                    continue

            cur_state = OrderedDict({})
            for name in state_names:
                cur_state[name] = self.cur_state[name][i]

            # uts.plot_images({'mask': cur_state['mask'],
            #                  'depth': cur_state['depth']})
            max_grad[i, :] = self.get_cur_grad(act, noises, cur_state)

        return mx.nd.array(max_grad)


    def update(self, data_batch):
        for (key, shape), value in zip(
                self.in_shape.items(), data_batch.data):
            self.actor.arg_dict[key][:] = value

        self.actor.forward(is_train=True)
        grad = self.get_gradient()
        self.actor.backward(grad)
        for i, pair in enumerate(zip(self.actor.arg_arrays, self.actor.grad_arrays)):
            weight, grad = pair
            self.actor_updater(i, grad, weight)


    def update_actor(self, obs, act, reward):
        for name in self.in_shape.keys():
            if name in obs.keys():
                self.actor.arg_dict[name][:] = obs[name]
            elif name in act.keys():
                self.actor.arg_dict[name][:] = act[name]
            elif name in reward.keys():
                self.actor.arg_dict[name][:] = reward[name]
            else:
                raise ValueError('miss %s for critic input' % name)

        # print 'input', obs['pose'], obs['crop']
        # print 'sampled pose:', act['del_pose']
        # print 'reward', reward['reward']

        count = 0
        while True:
            self.actor.forward(is_train=True)
            # for output in self.actor.outputs:
            #     print 'predict:', output.asnumpy()
            self.actor.backward()
        #     for i, pair in enumerate(zip(
        #              self.actor.arg_arrays, self.actor.grad_arrays)):
        #         self.actor_updater(i, self.actor.grad_arrays[i],
        #                 self.actor.arg_arrays[i])
            for i, index in enumerate(self.actor.grad_dict):
                if self.net_name in index and not (index in self.in_shape.keys()):
                    updater.adam_updater(self.actor.arg_dict[index],
                                         self.actor.grad_dict[index],
                                         self.actor_state[index],
                                         lr=self.c.actor_lr,
                                         wd=self.c.actor_wd)
            count += 1
            if count > 50:
                break

        # pdb.set_trace()


def train_pg(data_set='kitti'):
    """ gradient training to estimate relative pose
    """

    data_names = ['depth', 'mask', 'render_depth', 'init_pose']
    label_names = ['del_pose']
    params = data_libs[data_set].set_params_disp(disp='psm')

    # wrap the environment to data iter
    c.prefix = args.prefix
    params['batch_size'] = c.batch_size
    params['crop_size'] = [c.height, c.width]

    env = data_libs[data_set + '_env'].Env(c, params)
    setting = data_setting.get_policy_data_setting(env)
    train_iter = data_libs[data_set + '_iter'].EnvPolicyDataIter(
            params=params, env=env,
            setting=setting,
            data_names=data_names,
            label_names=label_names)
    agent = PG_trainer(c, env, train_iter, state_names=data_names)
    agent.init(init_model=args.init_model)
    agent.train()


def train_iter(dataset='kitti'):
    if dataset == 'kitti':
        import config.policy_config as c
    elif dataset == 'apollo':
        import config.policy_config_apollo as c

    c.is_crop = args.is_crop
    c.is_rel = args.is_rel
    ctx = [mx.gpu(int(i)) for i in args.gpu_id.split(',')]
    params = data_libs[dataset].set_params_disp(disp='psm')

    # wrap the environment to
    env = data_libs[dataset + '_env'].Env(c, params, split='train')

    params['batch_size'] = c.batch_size
    params['size'] = env.image_size
    params['crop_size'] = [c.height, c.width]

    if args.is_rel:
        data_names = ['image', 'depth', 'mask', 'pose', 'render_depth']
    if args.is_crop:
        data_names = ['image', 'depth', 'mask', 'crop', 'pose']

    label_names = ['del_pose']
    setting = data_setting.get_policy_data_setting(env)
    if args.is_sim:
        sampler = env.simulate_state
    else:
        sampler = env.sample_mcmc_state if dataset == 'kitti' else \
                env.sample_state_rect

    logging.info('init data iter train')
    train_data_iter = data_iter.PolicyDataIter(
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

    net_name = 'pose'
    outputs = []
    # use the geometric projection loss
    if c.is_crop:
        act_sym = pose_net.pose_block_w_crop(state, params, name=net_name,
                               is_rel=args.is_rel)
    else:
        act_sym = pose_net.pose_block(state, params, name=net_name,
                               is_rel=args.is_rel)

    loss = losses.my_pose_loss(act_sym[env.action_names[0]],
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


def train_resnet_iter(dataset, network='resnet'):
    if dataset == 'kitti':
        import config.policy_config as c
    elif dataset == 'apollo':
        import config.policy_config_apollo as c

    c.is_crop = args.is_crop
    c.is_rel = args.is_rel
    ctx = [mx.gpu(int(i)) for i in args.gpu_id.split(',')]
    params = data_libs[dataset].set_params_disp(disp='psm')

    # wrap the environment to
    env = data_libs[dataset + '_env'].Env(c, params, split='train')

    params['batch_size'] = c.batch_size
    params['size'] = env.image_size
    params['crop_size'] = [c.height, c.width]
    data_names = ['image', 'depth', 'mask', 'crop', 'pose']
    label_names = ['del_pose']

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

    net_name = 'actor'
    # use the geometric projection loss
    if network == 'resnet':
        act_sym = pose_net.resnet_pose_block(state, params,
                name=net_name, is_rel=args.is_rel)

    elif network == 'resnext':
        act_sym = pose_net.resnext_pose_block(state, params,
                name=net_name, is_rel=args.is_rel, )

    else:
        raise ValueError('no given network')

    outputs = []
    loss = losses.my_pose_loss(act_sym[label_names[0]],
                          action[label_names[0]],
                          batch_size=c.batch_size)
    outputs.append(loss)

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

    if args.is_pg:
        train_pg(args.dataset)
    else:
        if args.network == 'resnet' or args.network == 'resnext':
            if args.is_discret:
                train_resnet_iter_descrete(args.dataset, args.network)
            else:
                train_resnet_iter(args.dataset, args.network)
        elif args.network == 'demon':
            train_iter(args.dataset)
        else:
            raise ValueError(' No such network ')


