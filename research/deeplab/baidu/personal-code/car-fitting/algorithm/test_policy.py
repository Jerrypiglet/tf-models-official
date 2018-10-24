import mxnet as mx
import argparse
import data.data_setting as data_setting
import Networks.net_util as net_util
import data.kitti as kitti
import data.apolloscape as apollo
import data.kitti_env as kitti_env
import data.apolloscape_env as apollo_env
import logging
import network.car_pose_net as pose_net
import pickle as pkl
from collections import OrderedDict
import utils.utils as uts
import utils.metric as eval_metric
import data.kitti_iter as data_iter

data_libs = {}
data_libs['kitti'] = kitti
data_libs['apollo'] = apollo
data_libs['kitti_env'] = kitti_env
data_libs['apollo_env'] = apollo_env


def data_transform_4_network(state, state_names, data_transform):
    state_data = OrderedDict({})

    # for putting into hashing table
    state_data['image_name'] = state['image_name']
    state_data['inst_id'] = state['inst_id']
    for name in state_names:
        trans = data_transform[name]['transform']
        params = data_transform[name]['params'] if 'params' in \
                data_transform[name].keys() else {}
        if 'depth' in name:
            params = {'mean_depth': state['pose'][0, -1]}
        state_data[name] = trans(state[name], **params)
    return state_data



def test_object_pose(dataset='apollo'):
    """test a single network performance
    """
    if dataset == 'kitti':
        import config.policy_config as c
    elif dataset == 'apollo':
        import config.policy_config_apollo as c
    c.is_crop = True
    c.is_rel = False

    ctx = [mx.gpu(int(i)) for i in args.gpu_id.split(',')]
    params = data_libs[dataset].set_params_disp(disp='psm')
    env = data_libs[dataset + '_env'].Env(c, params, split='minival')

    data_transform = data_setting.get_policy_data_setting(env)
    sampler_eval = env.sample_mcmc_state if dataset == 'kitti' else \
                   env.sample_state_rect

    data_names = ['image', 'depth', 'mask', 'crop', 'pose']
    label_names = ['del_pose']
    in_size = [c.height, c.width]
    state_shapes = OrderedDict({
        "image": (c.batch_size, 3, in_size[0], in_size[1]),
        "depth": (c.batch_size, 1, in_size[0], in_size[1]),
        "render_depth": (c.batch_size, 1, in_size[0], in_size[1]),
        "mask": (c.batch_size, 1, in_size[0], in_size[1]),
        "pose": (c.batch_size, env.action_dim),
        "del_pose": (c.batch_size, env.action_dim),
        "reward": (c.batch_size, params['car_num'])})

    data_shapes = dict([(key, tuple([1] + list(shape[1:]))) for \
        key, shape in state_shapes.items()])

    sampler_val = env.sample_state_rect
    val_iter = data_iter.PolicyDataIter(
        params=params, env=env,
        sampler=sampler_val,
        setting=data_setting,
        data_names=data_names,
        label_names=label_names)

    net_name = 'pose'
    state = net_util.get_mx_var_by_name(data_names)
    net = pose_net.pose_block_w_crop(state, params, name=net_name,
                           is_rel=True)
    model = net_util.load_model({'model':args.test_model},
            data_names=data_names, data_shapes=data_shapes,
            net=net, ctx=ctx)

    pose_eval_metric = eval_metric.PoseMetric(
            output_names=['pose_output'],
            label_names=env.action_names, is_euler=True,
            trans_idx=[3, 4, 5], rot_idx=[0, 1, 2])

    car_poses = []
    while True:
        data_iter = iter(val_iter)
        try:
            data_batch = next(val_iter)
        except:
            break

        model.forward(data_batch.data)
        output = model.get_outputs()[0].asnumpy()
        pose_eval_metric.update(
                data_batch.label,
                output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a mx model.')
    parser.add_argument('--init_model', default=None,
        help='The type of fcn-xs model for init, e.g. fcnxs, fcn16s, fcn8s.')
    parser.add_argument('--gpu_id', default="2,3", help='the gpu ids use for training')
    parser.add_argument('--dataset', default="kitti",
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
    parser.add_argument('--network', default='resnet',
        help='policy net')
    parser.add_argument('--kv_store', type=str, default='device', help='the kvstore type')
    args = parser.parse_args()
    logging.info(args)

    if args.is_pg:
        train_pg(args.dataset)
    else:
        if args.network == 'resnet':
            train_resnet_iter(args.dataset)
        elif args.network == 'demon':
            train_iter(args.dataset)


