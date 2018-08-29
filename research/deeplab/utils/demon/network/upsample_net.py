import pdb

import paddle.v2 as pd
from collections import OrderedDict
from paddle.trainer_config_helpers.attrs import *
from demon_net import *
from layers.util_layers import get_cnn_input

__all__ = ['refine_block']


def get_inputs(params):
    inputs = {}
    inputs.update(get_cnn_input('image1', params['size'], 3))
    inputs.update(get_cnn_input('depth', params['size'], 1))
    inputs.update(get_cnn_input('depth_inv', params['size'], 1))
    return inputs


def get_ground_truth(params):
    gt = {}
    gt.update(get_cnn_input('depth_gt_inv', params['size'], 1))
    gt.update(get_cnn_input('depth_gt', params['size'], 1))
    gt.update(get_cnn_input('weight', params['size'], 1))
    return gt


# This file give a block for refinement of the upsampled results.
def refine_net(inputs, params, name='refine'):
    outputs = []
    image = pd.layer.concat(input=[inputs['image1'], inputs['depth']])
    conv0 = conv_bn_layer(image, 32, name=name + '_conv0')
    conv1 = conv_bn_layer(conv0, 64, stride=2, name=name + '_conv1')
    conv1_1 = conv_bn_layer(conv1, 64, name=name + '_conv1_1')
    conv2 = conv_bn_layer(conv1_1, 128, stride=2, name=name + '_conv2')
    conv2_1 = conv_bn_layer(conv2, 128, name=name + '_conv2_1')

    up_conv1 = deconv_bn_layer(conv2_1, 64, name=name + '_up_conv1')
    up_conv0 = pd.layer.concat(input=[up_conv1, conv1_1])
    up_conv0 = deconv_bn_layer(up_conv0, 32, name=name + '_up_conv0')
    up_conv0 = pd.layer.concat(input=[up_conv0, conv0])
    depth_feat = conv_bn_layer(up_conv0, 16, name=name + '_snet_conv1')
    depth_res = conv_bn_layer(depth_feat, 1,
                          act=pd.activation.Linear(),
                          name=name + '_snet_conv2')

    depth = pd.layer.addto(input=[depth_res, inputs['depth']],
                           act=pd.activation.Linear(),
                           bias_attr=False)
    depth_inv = pd.layer.mixed(input=[pd.layer.identity_projection(
                                  input=depth)],
                           act=pd.activation.Inv())
    outputs = OrderedDict(outputs)
    outputs.update(OrderedDict([('depth_inv',depth_inv),
                                ('depth', depth)]))

    return outputs


def upsample_net(inputs, params, name='upsample'):
    """Take in network output depth and sparse depth.
       Output the ground truth depth
    """

    pass