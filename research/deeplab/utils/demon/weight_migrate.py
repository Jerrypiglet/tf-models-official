# from tensorflow import weight, and save to paddle
import sys
paddle_root = '/home/peng/Paddle/'
sys.path.insert(0, paddle_root + 'python')
sys.path.insert(0, paddle_root + 'paddle/')

sys.path.insert(1, "./")

import pdb

import gzip
import cPickle as pkl
import numpy as np
import paddle.v2 as paddle
import data.sun3d as sun3d
import utils.utils as uts
import layers.cost_layers as cost_layers
import network.demon_net as d_net
from collections import OrderedDict
import argparse

def update(name_match, i, direct, prefix_tf, prefix_pd, suffix = '', sep = '_'):
    for name, name_b, p in zip(['kernel', 'bias'], ['', '.wbias'], ['', '_']):
        name_match[prefix_tf + i + suffix + direct + '/' + name] = \
                   p + prefix_pd + i + suffix + sep + direct + name_b;
    return name_match


def gen_demon_refine_block_name_matcher(name_match, net_name='refine_depth'):
    flow_scope = 'netRefine/'
    prefix_tf_flow = 'netRefine/conv'
    prefix_pd_flow = net_name + '_conv'

    for i in range(0, 3):
        update(name_match, str(i), '', prefix_tf_flow, prefix_pd_flow, sep='')
        if i > 0:
            update(name_match, str(i), '', prefix_tf_flow, \
            prefix_pd_flow, sep='', suffix='_1')

    # conv layers
    for i in range(0, 2):
        update(name_match, '', '', flow_scope + 'refine' + str(i) + '/upconv',
            net_name + '_up_conv', sep=str(i))

    for i in range(1, 3):
        update(name_match, str(i), '', flow_scope + 'predict_depth0/conv',
            net_name + '_snet_conv', sep='')

    return name_match


def gen_demon_depth_block_name_matcher(name_match, net_name='boost', is_iter=False):
    #load the tensorflow weights

    # first encoder decoder, conv layers
    if not is_iter:
        flow_scope = 'netDM1/'
        prefix_tf_flow = 'netDM1/conv'
        prefix_pd_flow = net_name + '_depth_block'
    else:
        flow_scope = 'netDM2/'
        prefix_tf_flow = 'netDM2/conv'
        prefix_pd_flow = net_name + '_depth_block'


    for i in range(1, 6):
        for direct in ['x', 'y']:
            update(name_match, str(i), direct, prefix_tf_flow, prefix_pd_flow)

        if i == 2:
            for direct in ['x', 'y']:
                update(name_match, str(i), direct, prefix_tf_flow, prefix_pd_flow,
                    suffix='_extra_inputs')
        if i > 1:
            for direct in ['x', 'y']:
                update(name_match, str(i), direct, prefix_tf_flow,\
                prefix_pd_flow, suffix='_1')

    # flow prediction layers
    update(name_match, '', '', flow_scope + 'motion_conv1',
        net_name + '_depth_motion', sep='')

    # conv layers
    for i in range(1, 4):
        update(name_match, '', '', flow_scope + 'motion_fc' + str(i),
            net_name + '_depth_fc' + str(i), sep='')

    # deconv layers
    for i in range(2, 5):
        update(name_match, '', '', flow_scope + 'refine' + str(i) + '/upconv',
            net_name + '_depth_up_block', sep=str(i))

    for i in range(1, 3):
        update(name_match, str(i), '', flow_scope + 'predict_depthnormal2/conv',
            net_name + '_depth_snet_up_conv', sep='')


    return name_match


def gen_demon_flow_block_name_matcher(name_match, net_name='boost', is_iter=False):
    #load the tensorflow weights
    def update(name_match, i, direct, prefix_tf, prefix_pd, suffix = '', sep = '_'):

        for name, name_b, p in zip(['kernel', 'bias'], ['', '.wbias'], ['', '_']):
            name_match[prefix_tf + i + suffix + direct + '/' + name] = \
                       p + prefix_pd + i + suffix + sep + direct + name_b;
        return name_match

    # first encoder decoder, conv layers
    if not is_iter:
        flow_scope = 'netFlow1/'
        prefix_tf_flow = 'netFlow1/conv'
        prefix_pd_flow = net_name + '_flow_block'
    else:
        flow_scope = 'netFlow2/'
        prefix_tf_flow = 'netFlow2/conv'
        prefix_pd_flow = net_name + '_flow_block'
        for direct in ['x', 'y']:
            update(name_match, str(2), direct, prefix_tf_flow,\
            prefix_pd_flow, suffix='_extra_inputs')

    for i in range(1, 6):
        for direct in ['x', 'y']:
            update(name_match, str(i), direct, prefix_tf_flow, prefix_pd_flow)
        if i > 1:
            for direct in ['x', 'y']:
                update(name_match, str(i), direct, prefix_tf_flow,\
                prefix_pd_flow, suffix='_1')


    for i in range(1, 3):
        update(name_match, str(i), '', flow_scope + 'predict_flow5/conv',
            net_name + '_flow_snet_conv', sep='')

    # flow prediction layers
    update(name_match, '', '', flow_scope + 'upsample_flow5to4/upconv',
        net_name + '_flow_up_flow', sep='')

    for i in range(2, 5):
        update(name_match, '', '', flow_scope + 'refine' + str(i) + '/upconv',
            net_name + '_flow_up_block', sep=str(i))

    for i in range(1, 3):
        update(name_match, str(i), '', flow_scope + 'predict_flow2/conv',
            net_name + '_flow_snet_up_conv', sep='')

    return name_match


def assign_weights(parameters, name_match, folder, model_name):
    np.set_printoptions(precision=20)
    for tf_name in name_match.keys():
    # for tf_name in ['netFlow1/conv1y/kernel']:
        with open(folder + tf_name.replace('/', '~') + '.pkl', 'rb') as f:
            weight = np.load(f)
        print "set {} : {}, {}, {}".format(tf_name, name_match[tf_name], \
            str(weight.shape), weight.dtype)

        if 'bias' in tf_name:
            weight = weight.reshape(parameters.get_shape(name_match[tf_name]))
            parameters.set(name_match[tf_name], weight)

        else:
            if len(weight.shape) == 4:
                # from [height,  width, in, out] to [out, in, height, width]
                weight = weight.transpose((3, 2, 0, 1))
                parameters.set(name_match[tf_name], weight.flatten().reshape((1, -1)))
            else:
                parameters.set(name_match[tf_name], weight)

    with gzip.open(model_name, 'w') as f:
        parameters.to_tar(f)


def main(argv):
    """
    main method of converting torch to paddle files.
    :param argv:
    :return:
    """
    cmdparser = argparse.ArgumentParser(
        "Convert tensorflow parameter file to paddle model files.")
    cmdparser.add_argument(
        '-i', '--input', help='input filename of torch parameters')
    cmdparser.add_argument('-l', '--layers', help='list of layer names')
    cmdparser.add_argument('-o', '--output', help='output file path of paddle model')

    args = cmdparser.parse_args(argv)

    params = sun3d.set_params('sun3d')
    params['stage'] = 5

    inputs = d_net.get_demon_inputs(params)

    # Add neural network config
    # outputs, out_field = d_net.get_demon_outputs(inputs, params, ext_inputs=inputs)
    outputs, out_field = d_net.get_demon_outputs(inputs, params, ext_inputs=None)

    # Create parameters
    parameters = paddle.parameters.create(outputs[out_field])
    # for name in parameters.names():
    #     print name

    name_match = OrderedDict([])
    if params['stage'] >= 1:
        name_match = gen_demon_flow_block_name_matcher(name_match)
    if params['stage'] >= 2:
        name_match = gen_demon_depth_block_name_matcher(name_match)
    if params['stage'] >= 3:
        # name_match = OrderedDict([])
        name_match = gen_demon_flow_block_name_matcher(name_match,
                                                       net_name='iter',
                                                       is_iter=True)
    if params['stage'] >= 4:
        name_match = gen_demon_depth_block_name_matcher(name_match,
                                                       net_name='iter',
                                                       is_iter=True)
    if params['stage'] >= 5:
        # name_match = OrderedDict([])
        name_match = gen_demon_refine_block_name_matcher(name_match)

    # for name in name_match.keys():
    #     print '{} : {}'.format(name, name_match[name])

    #Create depth paramters
    if not args.input:
        args.input = './output/tf_weights/'

    if not args.output:
        args.output = './output/tf_model_' + str(params['stage']) + '.tar.gz'

    print "save parameters to {}".format(args.output)
    assign_weights(parameters, name_match, args.input, args.output)


if __name__ == "__main__":
    main(sys.argv[1:])