"""Layers with parameters that need to be learnt
"""

import paddle.v2 as pd
from paddle.trainer_config_helpers.attrs import *

__all__ = ['conv_1d_layer',
           'conv_bn_layer',
           'deconv_bn_layer',
           'fc_bn_layer']


WITH_BN=False
DEFAULT_ACT=pd.activation.LeakyRelu()
init_std = None
init_mean = None

def conv_1d_layer(input,
                  num_filters,
                  filter_size=3,
                  stride=2,
                  act=DEFAULT_ACT,
                  with_bn=False,
                  name=None,
                  is_static=False,
                  ext=''):
    """1D swift of conv layers,

    Arguments:
      name: the name of parameter for share
    """
    assert(filter_size % 2 == 1)
    conv_name = name if name is None else name + '_y'
    param_attr = ParamAttr(name=conv_name,
                           initial_std=init_std,
                           initial_mean=init_mean,
                           is_static=is_static)

    conv1 = pd.layer.img_conv(input=input,
                              name=conv_name + ext,
                              num_filters=num_filters,
                              filter_size=1,
                              filter_size_y=filter_size,
                              stride=1,
                              stride_y=stride,
                              padding=0,
                              padding_y=(filter_size-1)/2,
                              bias_attr=False if with_bn else True,
                              act=act,
                              param_attr=param_attr)

    if with_bn:
        conv1 = pd.layer.batch_norm(input=conv1,
         act=pd.activation.LeakyRelu())

    conv_name = name if name is None else name + '_x'
    param_attr = ParamAttr(name=conv_name,
                           initial_std=init_std,
                           initial_mean=init_mean,
                           is_static=is_static)
    conv2 = pd.layer.img_conv(input=conv1,
                              name=conv_name + ext,
                              num_filters=num_filters,
                              filter_size=filter_size,
                              filter_size_y=1,
                              stride=stride,
                              stride_y=1,
                              padding=(filter_size-1)/2,
                              padding_y=0,
                              bias_attr=False if with_bn else True,
                              act=act,
                              param_attr=param_attr)


    if with_bn:
        conv2 = pd.layer.batch_norm(input=conv2,
          act=pd.activation.LeakyRelu())

    return conv2


def conv_bn_layer(input,
                  num_filters,
                  filter_size=3,
                  stride=1,
                  act=DEFAULT_ACT,
                  with_bn=False,
                  name=None,
                  is_static=False,
                  ext=''):

    param_attr = ParamAttr(name=name,
                           initial_std=init_std,
                           initial_mean=init_mean,
                           is_static=is_static)
    conv = pd.layer.img_conv(input=input,
                             name=name + ext,
                             num_filters=num_filters,
                             filter_size=filter_size,
                             stride=stride,
                             padding=(filter_size-1)/2,
                             bias_attr=False if with_bn else True,
                             act=act,
                             param_attr=param_attr)
    if with_bn:
        conv = pd.layer.batch_norm(input=conv,
          act=pd.activation.LeakyRelu())

    return conv


def deconv_bn_layer(input,
                    num_filters,
                    filter_size=4,
                    stride=2,
                    act=DEFAULT_ACT,
                    with_bn=False,
                    name=None,
                    is_static=False,
                    ext=''):

    param_attr = ParamAttr(name=name,
                           initial_std=init_std,
                           initial_mean=init_mean,
                           is_static=is_static)
    conv = pd.layer.img_conv(input=input,
                             name=name + ext,
                             num_filters=num_filters,
                             filter_size=filter_size,
                             stride=stride,
                             padding=(filter_size - 1) / 2,
                             bias_attr=False if with_bn else True,
                             act=act,
                             trans=True,
                             param_attr=param_attr)
    if with_bn:
        conv = pd.layer.batch_norm(input=conv,
            act=pd.activation.LeankyRelu())
    return conv


def fc_bn_layer(input,
                size,
                act=DEFAULT_ACT,
                with_bn=False,
                name=None,
                is_static=False,
                ext=''):

    param_attr = ParamAttr(name=name,
                           initial_std=init_std,
                           initial_mean=init_mean,
                           is_static=is_static)
    fc = pd.layer.fc(input=input,
                     name=name + ext,
                     size=size,
                     act=act,
                     bias_attr=False if with_bn else True,
                     param_attr=param_attr)
    if with_bn:
        fc = pd.layer.batch_norm(input=fc,
                                 act=pd.activation.LeakyRelu(),
                                 layer_attr=pd.attr.Extra(drop_rate=0.5))
    return fc

