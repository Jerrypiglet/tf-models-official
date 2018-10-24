""" Training script for pose regression
"""

import argparse
import mxnet as mx
import Networks.net_util as net_util
import Networks.mx_losses as losses
import network.car_pose_net as pose_net
import logging
import pdb

import numpy as np
import ddpg_car_fit as car_fit
import config.config as c
import data.kitti as kitti
import data.kitti_env as kitti_env
import data.kitti_iter as kitti_iter

np.set_printoptions(precision=3, suppress=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
# print os.environ['MXNET_ENGINE_TYPE']

data_libs = {}
data_libs['kitti'] = kitti
data_libs['kitti_env'] = kitti_env
data_libs['kitti_iter'] = kitti_iter

def train(data_set='kitti'):

    ctx = mx.gpu(int(args.gpu_ids))
    # ctx = mx.cpu()
    params = data_libs[data_set].set_params_disp()

    # wrap the environment to
    env = data_libs[data_set + '_env'].Env(c, params)
    agent = car_fit.DDPGCarFitting(env, c)
    train_iter = data_libs[data_set + '_iter'].EnvDataIter(
            params=params, env=env, agent=agent)

    # val_iter = data_libs[data_set + '_iter'].EnvDataIter(
            # params=params, env=env, agent=agent)
    print 'Init iter'
    checkpoint = mx.callback.do_checkpoint(args.prefix)
    log = mx.callback.Speedometer(1, 10)

    state = net_util.get_mx_var_by_name(env.state_names)
    action = net_util.get_mx_var_by_name(agent.action_names)
    label = net_util.get_mx_var_by_name(env.reward_names)

    net_name = 'critic'
    params['batch_size'] = c.batch_size
    params['size'] = env.image_size

    # use the geometric projection loss
    net = pose_net.value_net(state, action, params,
                             name=net_name)

    loss = 0
    for reward in env.reward_names:
        # loss += losses.euclidean_loss(net[net_name + '_' + reward],
        #                               label[reward],
        #                               batch_size=c.batch_size)
        loss += losses.smooth_l1_loss(net[net_name + '_' + reward],
                                      label[reward],
                                      batch_size=c.batch_size)
    print 'Init network'

    arg_params = None
    aux_params = None
    if args.init_model is not None:
        print 'loading model from existing network '
        models = {'model': args.init_model}
        symbol, arg_params, aux_params = net_util.args_from_models(models)

    mon = mx.monitor.Monitor(1)
    mod = mx.mod.Module(symbol=loss,
                        context=ctx,
                        data_names=env.state_names + agent.action_names,
                        label_names=env.reward_names)

    allow_missing = True if args.init_model is None else False
    optimizer_params = {'learning_rate':1e-5,
                        'wd':0.0005,
                        'clip_gradient': 1.0}
    # optimizer_params = {'learning_rate':0.0,
    #                     'wd':0.0000,
    #                     'clip_gradient': 0.5}
    mod.fit(train_iter,
            # eval_data=train_iter,
            # monitor=mon,
            optimizer='nadam',
            eval_metric='loss',
            optimizer_params=optimizer_params,
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=allow_missing,
            num_epoch=args.epoch,
            batch_end_callback=log,
            epoch_end_callback=checkpoint)


if __name__ == "__main__":
    # pdb.set_trace()
    parser = argparse.ArgumentParser(description='Train a mx model.')
    parser.add_argument('--init_model', default=None,
        help='The type of fcn-xs model for init, e.g. fcnxs, fcn16s, fcn8s.')
    parser.add_argument('--prefix', default='./output/value-net',
        help='The prefix(include path) of vgg16 model with mxnet format.')
    parser.add_argument('--epoch', type=int, default=150,
        help='The epoch number of vgg16 model.')
    parser.add_argument('--gpu_ids', default="3",
        help='the gpu ids use for training')
    parser.add_argument('--retrain', action='store_true', default=False,
        help='true means continue training.')
    args = parser.parse_args()
    logging.info(args)
    train()

