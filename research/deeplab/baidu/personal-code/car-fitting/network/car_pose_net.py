import logging
import math
import numpy as np

import mxnet as mx
import Networks.util_layers as utl
import Networks.resnet as resnet
import resnext
import Networks.net_util as net_util

from mxnet.ndarray import square
from mxnet.ndarray import sqrt

from collections import OrderedDict
import pdb


def resnet_pose_block(inputs,
                       data_params,
                       name='pose',
                       ext='',
                       ext_inputs=None,
                       arg_params=None,
                       is_rel=True):

    iter_name = '_pre'
    conv_bn_layer = utl.ConvBNLayer(ext=iter_name, params=arg_params)

    image = inputs['image']
    depth = inputs['depth']
    mask = inputs['mask']
    crop = inputs['crop']
    pose = inputs['pose']

    depth_mask = depth * mask
    image_feat = mx.symbol.concat(
            image, depth, depth_mask, mask, dim=1)

    h, w = data_params['crop_size']
    pose_dim = 10
    pose_feat = mx.sym.concat(crop, pose, dim=1)
    pose_feat = conv_embedding(pose_feat, pose_dim,
            6, [h, w], name, conv_bn_layer)
    in_source = mx.sym.concat(image_feat, pose_feat, dim=1)

    units = [3, 4, 6, 3]
    del_pose = resnet.resnet(data=in_source,
                    units=units,
                    num_stage=4,
                    filter_list=[64, 256, 512, 1024, 2048],
                    num_class=6,
                    bottle_neck = True,
                    bn_mom=0.9,
                    workspace=1024,
                    memonger=False)

    if is_rel:
        pose = del_pose + pose
    else:
        pose = del_pose

    Outputs = OrderedDict([('del_pose', pose)])

    return Outputs


def resnext_pose_block(inputs,
                       data_params,
                       name='pose',
                       ext='',
                       ext_inputs=None,
                       arg_params=None,
                       is_rel=True,
                       is_discrete=False,
                       bin_nums=None,
                       bin_vals=None,
                       bin_names=None):

    iter_name = '_pre'
    conv_bn_layer = utl.ConvBNLayer(ext=iter_name, params=arg_params)

    image = inputs['image']
    depth = inputs['depth']
    mask = inputs['mask']
    crop = inputs['crop']
    pose = inputs['pose']

    depth_mask = depth * mask
    image_feat = mx.symbol.concat(
            image, depth, depth_mask, mask, dim=1)

    h, w = data_params['crop_size']
    pose_dim = 10
    pose_feat = mx.sym.concat(crop, pose, dim=1)
    pose_feat = conv_embedding(pose_feat, pose_dim,
            6, [h, w], name, conv_bn_layer)
    in_source = mx.sym.concat(image_feat, pose_feat, dim=1)

    units = [3, 4, 6, 3]
    num_group = 32
    in_channel = 12

    if is_discrete:
        disc_pose = resnext.resnext(data=in_source,
                       units=units,
                       num_stages=4,
                       filter_list=[64, 256, 512, 1024, 2048],
                       num_classes=bin_nums,
                       num_group=num_group,
                       bottle_neck=True,
                       image_shape=[in_channel, h, w],
                       domain=bin_names,
                       workspace=1024)

        del_pose = net_util.arg_softmax(disc_pose, bin_vals, \
                data_params['batch_size'])

        pose = del_pose + pose if is_rel else del_pose
        Outputs = OrderedDict([('del_pose', pose),
            ('disc_pose', disc_pose)])
        return Outputs

    else:
        del_pose = resnext.resnext(data=in_source,
                       units=units,
                       num_stages=4,
                       filter_list=[64, 256, 512, 1024, 2048],
                       num_classes=6,
                       num_group=num_group,
                       bottle_neck=True,
                       image_shape=[in_channel, h, w],
                       workspace=1024)

        pose = del_pose + pose if is_rel else del_pose
        Outputs = OrderedDict([('del_pose', pose)])
        return Outputs


def pose_block_w_crop(inputs,
               data_params,
               name='pose',
               ext='',
               ext_inputs=None,
               iter_num=None,
               arg_params=None,
               is_rel=True):
    """ inputs ->
    """

    iter_name = '' if iter_num is None else '_' + str(iter_num)
    iter_name = iter_name + ext

    fc_bn_layer = utl.FCBNLayer(ext=iter_name, params=arg_params)
    conv_bn_layer = utl.ConvBNLayer(ext=iter_name, params=arg_params)
    activation = 'PRELU'

    image = inputs['image']
    depth = inputs['depth']
    mask = inputs['mask']
    crop = inputs['crop']
    pose = inputs['pose']

    # trans = mx.sym.slice_axis(inputs['pose'], axis=1, begin=3, end=6)
    # crop 4 pose 6
    pose_feat = mx.sym.concat(crop, pose, dim=1)
    h, w = data_params['crop_size']
    pose_dim = 10
    pose_feat = conv_embedding(pose_feat, pose_dim,
            16, [h, w], name, conv_bn_layer)
    # pose_feat = fc_bn_layer(pose_feat, h * w)
    # pose_feat = mx.sym.reshape(pose_feat, (-1, 1, h, w))

    depth_mask = depth * mask
    image_feat = mx.symbol.concat(
            image, depth, depth_mask, mask, dim=1)
    image_feat = conv_bn_layer(
            image_feat, 16, 3, 1, name=name + '_conv1',
            act=activation)

    in_source = mx.sym.concat(image_feat, pose_feat, dim=1)
    conv_feat = utl.conv_block(in_source,
            arg_params=arg_params, name=name, ext=iter_name,
            act=activation)

    motion_feat = conv_bn_layer(
            conv_feat['block5_1'], 128, 3, 1, name=name + '_motion',
            act=activation)
    fc1 = fc_bn_layer(motion_feat, 1024, name=name + '_fc_1',
            act=activation)
    fc2 = fc_bn_layer(fc1, 128, name=name + '_fc_2', act=activation)
    del_pose = fc_bn_layer(fc2, 6, act=None, name=name + '_pose')

    if is_rel:
        pose = pose + del_pose
    else:
        pose = del_pose

    Outputs = OrderedDict([('del_pose', pose)])
    # car_id = fc_bn_layer(fc1, data_params['car_num'], act=None, name=name + '_car_id')
    # pose = pose + inputs['pose']
    # Outputs = OrderedDict([('pose', pose), ('car_id', car_id)])
    return Outputs


def flat_embedding(in_pose, sz, name, fc_bn_layer):
    h, w = sz
    pose_f = fc_bn_layer(in_pose, 1, name=name + '_embed', with_bn=False)
    pose_f = mx.sym.tile(pose_f, reps=(1, h * w))
    pose_f = mx.sym.reshape(pose_f, (-1, 1, h, w))

    return pose_f


def conv_embedding(in_pose, in_dim, ch, sz, name, conv_bn_layer):
    h, w, = sz
    pose_f = mx.sym.tile(in_pose, reps=(1, h*w))
    pose_f = mx.sym.reshape(pose_f, (-1, h, w, in_dim))
    pose_f = mx.sym.transpose(pose_f, (0, 3, 1, 2))
    pose_f = conv_bn_layer(
        pose_f, ch, 1, 1, name=name + '_embedding', act=None)

    return pose_f


def pose_block(inputs,
               data_params,
               name='pose',
               ext='',
               ext_inputs=None,
               iter_num=None,
               arg_params=None,
               is_rel=False):
    """ inputs ->
    """

    iter_name = '' if iter_num is None else '_' + str(iter_num)
    iter_name = iter_name + ext

    fc_bn_layer = utl.FCBNLayer(ext=iter_name, params=arg_params)
    conv_bn_layer = utl.ConvBNLayer(ext=iter_name, params=arg_params)
    activation = 'RELU'

    # image = inputs['image']
    depth = inputs['depth']
    mask = inputs['mask']
    depth_mask = depth * mask
    # pose = inputs['pose']
    # h, w = data_params['size']

    if is_rel:
        render_depth = inputs['render_depth']
        # in_pose = inputs['pose'] / 10.0
        h, w = data_params['size']
        # pose_f = mx.sym.tile(in_pose, reps=(1, h*w))
        # pose_f = mx.sym.reshape(pose_f, (-1, h, w, 6))
        # pose_f = mx.sym.transpose(pose_f, (0, 3, 1, 2))
        # pose_f = conv_bn_layer(
        #     pose_f, 1, 1, 1, name=name + '_embedding', act=None)
        # pose_f = flat_embedding(in_pose, [h, w], name, fc_bn_layer)
        in_source = mx.symbol.concat(
                     depth, depth_mask, render_depth, render_depth, dim=1)
    else:
        in_source = mx.symbol.concat(
                     depth_mask, depth_mask, mask, mask, dim=1)

    conv_feat = utl.conv_block(in_source,
            arg_params=arg_params, name=name, ext=iter_name,
            act=activation)
    motion_feat = conv_bn_layer(
            conv_feat['block5_1'], 128, 3, 1, name=name + '_motion',
            act=activation)
    fc1 = fc_bn_layer(motion_feat, 1024, name=name + '_fc_1',
            act=activation)
    fc2 = fc_bn_layer(fc1, 128, name=name + '_fc_2', act=activation)
    pose = fc_bn_layer(fc2, 6, act=None, name=name + '_pose')

    if is_rel:
        pose = pose + inputs['pose']

    Outputs = OrderedDict([('del_pose', pose)])

    # car_id = fc_bn_layer(fc1, data_params['car_num'], act=None, name=name + '_car_id')
    # pose = pose + inputs['pose']
    # Outputs = OrderedDict([('pose', pose), ('car_id', car_id)])
    return Outputs

def seg_net():


def value_net(obs,
              act,
              data_params,
              arg_params=None,
              name='pose',
              ext=''):
    alpha = 0.4

    fc_bn_layer = utl.FCBNLayer(params=arg_params)
    conv_bn_layer = utl.ConvBNLayer(params=arg_params)
    activation = 'PRELU'

    # image = obs['image']
    depth = obs['depth']
    mask = obs['mask']
    render_depth = obs['render_depth']
    h, w = data_params['size']
    in_pose = (act['del_pose'] + obs['pose']) / 10.0

    # pose_f = mx.sym.tile(pose, reps=(1, h*w))
    # pose_f = mx.sym.reshape(pose_f, (data_params['batch_size'],
    #     h, w, -1))
    pose_f = flat_embedding(in_pose, [h, w], name, fc_bn_layer)

    # in_source = mx.symbol.concat(
    #                image, depth, mask, pose_f, dim=1)
    depth_mask = depth * mask
    in_source = mx.symbol.concat(
                    depth, depth_mask, render_depth, pose_f, dim=1)

    conv_feat = utl.conv_block(in_source,
                               arg_params=arg_params,
                               name=name,
                               ext=ext,
                               act=activation)
    # motion_feat = conv_bn_layer(
    #         in_source, 64, 3, 1, name=name + '_c1', act='LeakyRELU')
    # motion_feat = conv_bn_layer(
    #         motion_feat, data_params['car_num'], 3, 1, name=name + '_motion', act='LeakyRELU')

    # fc1 = conv_bn_layer(motion_feat, 1024, name=name + '_fc1')
    # fc2 = fc_bn_layer(fc1, 128, name=name + '_fc2')
    # motion_feat = conv_bn_layer(
    #         conv_feat['block5_1'], data_params['car_num'], 3, 1,
    #         name=name + '_motion')
    # reward = mx.sym.Pooling(
    #         motion_feat, global_pool=True, pool_type='avg', kernel=(10, 10))
    # reward = mx.sym.reshape(reward, (data_params['batch_size'], -1))

    motion_feat = conv_bn_layer(
            conv_feat['block5_1'], 128, 3, 1, name=name + '_motion',
            act=activation)
    fc1 = fc_bn_layer(motion_feat, 1024, name=name + '_fc_1', act=activation)
    fc2 = fc_bn_layer(fc1, 128, name=name + '_fc_2', act=activation)

    # IOU = fc_bn_layer(fc2, data_params['car_num'], act=None,
    #         name=name + '_IOU')
    # rel = fc_bn_layer(fc2, data_params['car_num'], act=None,
    #         name=name + '_rel')
    reward = fc_bn_layer(fc2, data_params['car_num'], act=None,
            name=name + '_reward')
    # reward = IOU + alpha * rel
    # car_id = mx.sym.argmax(reward, axis=1)

    # Outputs = OrderedDict([(name + '_IOU' , IOU),
    #                        (name + '_rel', rel),
    #                        (name + '_reward', reward),
    #                        (name + '_car_id', car_id)])
    Outputs = OrderedDict([(name + '_reward', reward)])

    return Outputs


class DDPGNet(object):
    """
    Continous Multi-Layer Perceptron Q-Value Network
    for determnistic policy training.
    """
    def __init__(self, c, env, agent, init_model=None):
        """ c is the configuration for learning
            data_params is the parameters for datasets
        """

        # self.ctx = [mx.gpu(int(i)) for i in c.gpu_ids.split(',')]
        if c.gpu_flag:
            self.ctx = mx.gpu(int(c.gpu_id))
        else:
            self.ctx = mx.cpu()

        self.env = env
        self.agent = agent
        self.init_model = init_model
        self.config = c
        self.params = {}
        self.params['batch_size'] = c.batch_size
        self.params['size'] = env.image_size
        self.params['car_num'] = env.data_params['car_num']

        in_size = [c.height, c.width] if c.is_crop else \
                env.image_size

        self.state_shapes = OrderedDict({
            "image": (c.batch_size, 3, in_size[0], in_size[1]),
            "depth": (c.batch_size, 1, in_size[0], in_size[1]),
            "render_depth": (c.batch_size, 1, in_size[0], in_size[1]),
            "mask": (c.batch_size, 1, in_size[0], in_size[1]),
            "pose": (c.batch_size, self.env.action_dim),
            "del_pose": (c.batch_size, self.env.action_dim),
            "reward": (c.batch_size, self.params['car_num'])})

        self.one_state_shapes = dict([(key,
            tuple([1] + list(shape[1:]))) for \
            key, shape in self.state_shapes.items()])

        self.batch_size = c.batch_size
        self.in_size = np.round(env.image_size)
        self.action_dim = agent.action_dim
        self.data_params = env.data_params

        obs_names = ['depth', 'mask', 'pose', 'render_depth']
        act_names = ['del_pose']

        self.actor_in_names = obs_names
        self.critic_in_names = obs_names + act_names + \
            self.env.reward_names

        self.obs = net_util.get_mx_var_by_name(obs_names)
        self.act = net_util.get_mx_var_by_name(agent.action_names)
        self.reward = net_util.get_mx_var_by_name(env.reward_names)


    def create_actor_net(self, obs):
        # pose = obs['pose']
        # car_id = obs['car_id']

        # vertices = mx.sym.take(self.vertices, car_id)
        # faces = mx.sym.take(self.faces, car_id)

        # render_depth, render_mask = mx.symbol.Custom(
        #         pose=pose,
        #         vertices=vertices,
        #         faces=faces,
        #         name='proj_3d_car',
        #         op_type='proj_3d_car')

        # obs['render_depth'] = render_depth
        # obs['render_mask'] = render_mask
        act = pose_block(obs,
                       data_params=self.params,
                       name='actor',
                       ext='',
                       ext_inputs=None,
                       iter_num=None,
                       arg_params=None)

        return act


    def create_critic_net(self, obs, act):

        val = value_net(obs,
                        act,
                        data_params=self.params,
                        arg_params=None,
                        name='critic',
                        ext='')

        return val


    def critic_loss(self, qval_sym_critic, c_name):
        critic_loss = 1.0 / self.batch_size * mx.symbol.sum(
           mx.symbol.square(qval_sym_critic[c_name] - \
                       self.reward[self.env.reward_names[0]]))

        critic_loss = mx.symbol.MakeLoss(critic_loss, name="critic_loss")
        critic_out = mx.sym.Group([critic_loss, mx.sym.BlockGrad(qval_sym_critic[c_name])])

        return critic_out


    def init(self):
        # Critic initilization
        c_name = 'critic_reward'
        a_name = 'del_pose'

        qval_sym_critic = self.create_critic_net(self.obs, self.act)
        critic_out = self.critic_loss(qval_sym_critic, c_name)


        critic_input_shapes = OrderedDict(zip(self.critic_in_names, \
                [self.state_shapes[name] for name in \
                self.critic_in_names]))
        self.critic = critic_out.simple_bind(
                ctx=self.ctx, **critic_input_shapes)

        # for saving network symbols
        self._critic_out = critic_out

        # maximize reward
        grad_loss = -10 * mx.sym.sum(mx.sym.max(
            qval_sym_critic[c_name], axis=1)) / self.batch_size

        grad_arg_dict = {}
        for name, arr in self.critic.arg_dict.items():
            if name not in ['reward']:
                grad_arg_dict[name] = arr

        # self.grad_batch = mx.nd.empty((self.batch_size, self.action_dim),
        #                                ctx=self.ctx)
        # print grad_loss.list_arguments()

        # self.grad = mx.symbol.MakeLoss(grad_loss).bind(
        #                         ctx=self.ctx, args=grad_arg_dict,
        #                         args_grad={a_name: self.grad_batch},
        #                         shared_exec=self.critic)
        args_grad = {}
        for name, arr in self.critic.grad_dict.items():
        	if name not in ["reward"]:
        		args_grad[name] = mx.nd.empty(arr.shape, ctx=self.ctx)

        self.grad = mx.symbol.MakeLoss(grad_loss, name='reward_loss').bind(
            ctx=self.ctx,
        	args=grad_arg_dict,
            args_grad=args_grad,
            grad_req="write")

        # Actor initilization
        act_sym = self.create_actor_net(self.obs)

        actor_in_shapes = OrderedDict(zip(self.actor_in_names, \
                [self.state_shapes[name] for name in self.actor_in_names]))
        self.actor = act_sym[a_name].simple_bind(
                       ctx=self.ctx, **actor_in_shapes)
        self._actor_sym = act_sym[a_name]

        actor_one_in_shapes = OrderedDict(zip(self.actor_in_names, \
                [self.one_state_shapes[name] for name in \
                self.actor_in_names]))
        self.actor_one = self.actor.reshape(**actor_one_in_shapes)

        # the whole network
        qval_sym_actor = self.create_critic_net(self.obs, act_sym)
        target_out = mx.sym.Group([qval_sym_actor[c_name],
            act_sym[a_name]])
        self.target = target_out.simple_bind(ctx=self.ctx,
                **actor_in_shapes)

        # define optimizer
        self.critic_updater = mx.optimizer.get_updater(mx.optimizer.create(
            self.config.critic_updater,
            learning_rate=self.config.critic_lr,
            wd=self.config.weight_decay))
        self.actor_updater = mx.optimizer.get_updater(mx.optimizer.create(
            self.config.actor_updater, learning_rate=self.config.actor_lr))

        # init params
        self.init_params()


    def init_params(self):
        for name, arr in self.target.arg_dict.items():
            initializer = mx.initializer.Uniform(
                    self.config.init_scale)
            initializer._init_weight(name, arr)

        # use for update parameters
        self.critic_state = {}
        self.actor_state = {}
        for name, arr in self.target.arg_dict.items():
            if 'actor' in name:
                arr.copyto(self.actor.arg_dict[name])
                shape = self.actor.arg_dict[name].shape
                self.actor_state[name] = (mx.nd.zeros(shape, self.ctx),
                        mx.nd.zeros(shape, self.ctx))

            if 'critic' in name:
                arr.copyto(self.critic.arg_dict[name])
                shape = self.critic.arg_dict[name].shape
                self.critic_state[name] = (mx.nd.zeros(shape, self.ctx),
                        mx.nd.zeros(shape, self.ctx))


    def update_critic(self, obs, act, reward):
        for name in self.critic_in_names:
            if name in obs.keys():
                self.critic.arg_dict[name][:] = obs[name]
            elif name in act.keys():
                self.critic.arg_dict[name][:] = act[name]
            elif name in reward.keys():
                self.critic.arg_dict[name][:] = reward[name]
            else:
                raise ValueError('miss %s for critic input' % name)

        self.critic.forward(is_train=True)
        self.critic.backward()

        # print "critic param :", self.critic.arg_dict['critic_IOU_bias'].asnumpy()
        # print "critic grad :", self.critic.grad_dict['critic_IOU_bias'].asnumpy()
        for i, index in enumerate(self.critic.grad_dict):
            if 'critic' in index:
                self.update_param(self.critic.arg_dict[index],
                                  self.critic.grad_dict[index],
                                  self.critic_state[index],
                                  lr=self.config.critic_lr,
                                  wd=self.config.weight_decay)
                # self.critic.arg_dict[index].copyto(self.grad.arg_dict[index])

        # print "updating ................."
        # print "critic param :", self.critic.arg_dict['critic_motion_bias'].asnumpy()
        # print "critic grad :", self.critic.grad_dict['critic_motion_bias'].asnumpy()
        # print "step updating over.............................."


    def update_actor(self, obs):
        # print 'obs keys %s' % (obs.keys())
        for name in self.actor_in_names:
            self.actor.arg_dict[name][:] = obs[name]

        self.actor.forward(is_train=True)

        # for getting gradient for action
        for name, value in obs.items():
            self.grad.arg_dict[name][:] = value

        act_name = self.agent.action_names[0]
        self.grad.arg_dict[act_name][:] = self.actor.outputs[0]

        self.grad.forward()
        self.grad.backward()

        # print 'grad loss %s' % self.grad.outputs[0].asnumpy()
        # pdb.set_trace()
        # print self.grad.grad_dict[act_name].asnumpy()
        # print self.grad.grad_dict[act_name].asnumpy()

        self.actor.backward(self.grad.grad_dict['del_pose'])
        for i, index in enumerate(self.actor.arg_dict):
            if 'actor' in index:
                self.update_param(
                        self.actor.arg_dict[index],
                        self.actor.grad_dict[index],
                        self.actor_state[index], lr=self.config.actor_lr)

        # self.actor.arg_dict[index].copyto(self.actor_one.arg_dict[index])
        # self.actor.forward(is_train=True)
        # print "actor loss after update", self.actor.outputs[0].asnumpy()

    def update_target(self):
        for name, arr in self.target.arg_dict.items():
            if 'actor' in name:
                self.target.arg_dict[name] = (1 - self.config.soft_target_tau)* arr + \
                        self.config.soft_target_tau * self.actor.arg_dict[name]

            elif 'critic' in name:
                self.target.arg_dict[name] = (1 - self.config.soft_target_tau)* arr + \
                        self.config.soft_target_tau * self.critic.arg_dict[name]


    def get_target_q(self, obs):
        for name, value in obs.items():
            self.target.arg_dict[name][:] = value

        self.target.forward(is_train=False)
        return self.target.outputs[0]


    def get_step_action(self, obs):
        # single observation
        for name, value in obs.items():
            self.actor_one.arg_dict[name][:] = value

        self.actor_one.forward(is_train=False)
        return self.actor_one.outputs[0].asnumpy()


    def update_param(self, weight, grad, state,
            lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, wd=0):

        mean, variance = state

        mean *= beta1
        mean += grad * (1. - beta1)

        variance *= beta2
        variance += (1 - beta2) * square(grad)

        coef1 = 1. - beta1
        coef2 = 1. - beta2
        lr *= math.sqrt(coef2) / coef1

        weight -= lr * mean / (sqrt(variance) + epsilon)
        if wd > 0.:
            weight[:] -= (lr * wd) * weight


    def save_networks(self, netname, prefix, epoch):
        param_name = '%s-%s-%04d.params' % (prefix, netname, epoch)
        if netname == 'actor':
            self._actor_sym.save('%s-%s-symbol.json' % (prefix, netname))
            save_dict = { k : v.as_in_context(mx.cpu()) \
                          for k, v in self.actor.arg_dict.items()}
        elif netname == 'critic':
            self._critic_out.save('%s-%s-symbol.json' % (prefix, netname))
            save_dict = { k : v.as_in_context(mx.cpu()) \
                          for k, v in self.critic.arg_dict.items()}
        else:
            raise ValueError('no given network %s \n' % netname)

        mx.ndarray.save(param_name, save_dict)
        logging.info('Save checkpoint to %s' % param_name)


    def load_networks(self, netname, model_name, allow_missing=True):

        arg_params, _ = net_util.load_mxparams_from_file(model_name + '.params')
        if netname == 'actor':
            for name, arr in self.actor.arg_dict.items():
                if netname in name:
                    if not (name in arg_params.keys()):
                        if not allow_missing:
                            raise ValueError('%s is missing' % name)
                        else:
                            continue
                    self.actor.arg_dict[name][:] = arg_params[name]
                    self.actor.arg_dict[name].copyto(
                            self.target.arg_dict[name])

        elif netname == 'critic':
            for name, arr in self.critic.arg_dict.items():
                if netname in name:
                    if not (name in arg_params.keys()):
                        if not allow_missing:
                            raise ValueError('%s is missing' % name)
                        else:
                            continue
                    self.critic.arg_dict[name][:] = arg_params[name]
                    self.critic.arg_dict[name].copyto(
                            self.target.arg_dict[name])
        else:
            raise ValueError('no given network %s \n' % netname)

        logging.info('Load checkpoint to %s' % model_name)


if __name__ == '__main__':
    d=DDPGNet(4,1)
    d.init()

