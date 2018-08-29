""" Include all specific setting for the network
"""

import pdb

import paddle.v2 as pd
from collections import OrderedDict
from paddle.trainer_config_helpers.attrs import *
from layers.util_layers import get_cnn_input

__all__ = ['conv_1d_layer',
           'conv_bn_layer',
           'deconv_bn_layer',
           'fc_bn_layer']

WITH_BN=False
DEFAULT_ACT=pd.activation.LeakyRelu()
# init_std = 0.0001
# init_mean = 0.
init_std = None
init_mean = None
# DEFAULT_ACT=pd.activation.Relu()


class ConvBlockBase(object):
  def __init__():
    pass


class Conv1DBlock(ConvBlockBase):
  def __init__():
    pass


def get_demon_inputs(params, stage='boostrap'):
    inputs = {}
    inputs.update(get_cnn_input('image1', params['size'], 3))
    inputs.update(get_cnn_input('image2', params['size'], 3))
    image2_down = pd.layer.bilinear_interp(input=inputs['image2'],
                              out_size_x=params['size_stage'][1][1],
                              out_size_y=params['size_stage'][1][0])
    inputs.update({'image2_down': image2_down})
    inputs['intrinsic'] = pd.layer.data(
        name="intrinsic", type=pd.data_type.dense_vector(4))

    inputs.update(get_cnn_input('depth_inv', params['size_stage'][1], 1))
    inputs.update(get_cnn_input('normal', params['size_stage'][1], 3))

    inputs['rotation'] = pd.layer.data(
        name="rotation", type=pd.data_type.dense_vector(3))
    inputs['translation'] = pd.layer.data(
        name="translation", type=pd.data_type.dense_vector(3))

    return inputs


def get_ground_truth(params):
    gt = {}
    gt.update(get_cnn_input('weight', params['size'], 1))
    gt.update(get_cnn_input('flow_gt', params['size'], 2))
    gt.update(get_cnn_input('depth_gt', params['size'], 1))
    gt.update(get_cnn_input('normal_gt', params['size'], 3))

    gt['rotation_gt'] = pd.layer.data(
        name="rotation_gt", type=pd.data_type.dense_vector(3))
    gt['translation_gt'] = pd.layer.data(
        name="translation_gt", type=pd.data_type.dense_vector(3))
    return gt


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



def flow_block(inputs, params, name='boost_flow', ext_inputs=None, iter='',
               is_static=False):
    outputs = []
    # separate inputs and ext_inputs is because inputs is global level input
    # from data and ext_inputs from output in previous stages

    pair = pd.layer.concat(input=[inputs['image1'], inputs['image2']])
    block1 = conv_1d_layer(pair, 32, 9, name=name + '_block1', ext=iter)

    if name == 'iter_flow':
      block2 = conv_1d_layer(block1, 32, 7, name=name + '_block2', ext=iter)
      trans = pd.layer.concat(input=[inputs['intrinsic'],
                                     ext_inputs['rotation'],
                                     ext_inputs['translation']])
      depth = pd.layer.mixed(input=[pd.layer.identity_projection(
                                    input=ext_inputs['depth_inv'])],
                             act=pd.activation.Inv())
      flow_trans = pd.layer.trans_depth_flow(input=[depth, trans],
                                             depth2flow=True)
      image_warp = pd.layer.warp2d(input=[inputs['image2_down'], flow_trans])
      ex_input = pd.layer.concat(input=[image_warp, flow_trans,
                                        ext_inputs['depth_inv'],
                                        ext_inputs['normal']],
                                 name='concat_iter_flow' + iter)

      block2_extra_inputs = conv_1d_layer(ex_input, 32, 3, 1,
                              name=name + '_block2_extra_inputs', ext=iter)
      outputs.append(('block2_ext', block2_extra_inputs))
      block2 = pd.layer.concat(input=[block2, block2_extra_inputs],
                               name='concat_iter_flow_block2' + iter)

    else:
      block2 = conv_1d_layer(block1, 64, 7, name=name + '_block2',ext=iter)

    block2_1 = conv_1d_layer(block2, 64, 3, 1, name=name + '_block2_1',ext=iter)
    block3 = conv_1d_layer(block2_1, 128, 5, name=name + '_block3',ext=iter)
    block3_1 = conv_1d_layer(block3, 128, 3, 1, name=name + '_block3_1',ext=iter)
    block4 = conv_1d_layer(block3_1, 256, 5, name=name + '_block4',ext=iter)
    block4_1 = conv_1d_layer(block4, 256, 3, 1, name=name + '_block4_1',ext=iter)
    block5 = conv_1d_layer(block4_1, 512, 5, name=name + '_block5',ext=iter)
    block5_1 = conv_1d_layer(block5, 512, 3, 1, name=name + '_block5_1',ext=iter)

    for i in range(1, 6):
      exec('outputs.append((\'block' + str(i) +'\', block' + str(i) + '))')
      if i > 1:
        exec('outputs.append((\'block' + str(i) +'_1\', block' + str(i) + '_1))')

    code_flow = conv_bn_layer(block5_1, 24, name=name + '_snet_conv1',ext=iter)
    outputs.append(('code_flow', code_flow))

    flow_out = conv_bn_layer(code_flow, 4, 3, 1,
                         pd.activation.Linear(), False,
                         name=name + '_snet_conv2',ext=iter)
    # flow_out = conv_bn_layer(code_flow, 4, name=name + '_snet_conv2')
    outputs.append(('flow_out_low', flow_out))

    flow_low = pd.layer.slice(input=flow_out,begin=0,size=2,axis=1)
    flow_conf_low = pd.layer.slice(input=flow_out,begin=2,size=2,axis=1)

    up_flow = deconv_bn_layer(flow_out, 2, 4, 2,
                              pd.activation.Linear(), False,
                              name=name + '_up_flow',ext=iter)
    outputs.append(('up_flow', up_flow))

    up_block4 = deconv_bn_layer(block5_1, 256, 4, name=name + '_up_block4',ext=iter)
    up_block4 = pd.layer.concat(input=[up_block4, block4_1, up_flow])
    up_block3 = deconv_bn_layer(up_block4, 128, 4, name=name + '_up_block3',ext=iter)
    up_block3 = pd.layer.concat(input=[up_block3, block3_1])
    up_block2 = deconv_bn_layer(up_block3, 64, 4, name=name + '_up_block2',ext=iter)
    up_block2 = pd.layer.concat(input=[up_block2, block2_1])
    code_flow_up = conv_bn_layer(up_block2, 24, name=name + '_snet_up_conv1',ext=iter)
    flow_out_up = conv_bn_layer(code_flow_up, 4, 3, 1,
                            pd.activation.Linear(), False,
                            name=name + '_snet_up_conv2',ext=iter)

    for i in range(2, 5)[::-1]:
      exec('outputs.append((\'up_block' + str(i) +'\', up_block' + str(i) + '))')

    outputs.append(('flow_out_up', flow_out_up))

    # pdb.set_trace()
    flow_up = pd.layer.slice(input=flow_out_up, begin=0, size=2, axis=1)
    flow_conf_up = pd.layer.slice(input=flow_out_up,begin=2,size=2,axis=1)

    # outputs = OrderedDict(outputs)
    outputs = OrderedDict([('flow_low', flow_low),
                           ('flow_conf_low', flow_conf_low),
                           ('flow_out', flow_out_up),
                           ('flow', flow_up),
                           ('flow_conf', flow_conf_up)])
    return outputs


def depth_block(inputs, flow_out, params, name='boost_depth', ext_inputs=None, iter=''):
    """block predicting depth from the output from flow
    """
    outputs = []
    pair = pd.layer.concat(input=[inputs['image1'], inputs['image2']])

    block1 = conv_1d_layer(pair, 32, 9, name=name + '_block1',ext=iter)
    block2 = conv_1d_layer(block1, 32, 7, name=name + '_block2',ext=iter)

    image_warp = pd.layer.warp2d(input=[inputs['image2_down'], flow_out['flow']])

    if name == 'iter_depth':
      trans = pd.layer.concat(input=[inputs['intrinsic'],
                                     ext_inputs['rotation'],
                                     ext_inputs['translation']])

      depth_trans = pd.layer.trans_depth_flow(input=[flow_out['flow'], trans],
                                              depth2flow=False)
      depth_trans = pd.layer.mixed(input=[pd.layer.identity_projection(
                                          input=depth_trans)],
                                   act=pd.activation.Inv())
      input_s2 = pd.layer.concat(input=[image_warp,
                                        flow_out['flow_out'],
                                        depth_trans])

    else:
      input_s2 = pd.layer.concat(input=[image_warp, flow_out['flow_out']])

    block2_ex = conv_1d_layer(input_s2, 32, 3, 1,
       name=name + '_block2_extra_inputs',ext=iter)

    block2_concat = pd.layer.concat(input=[block2, block2_ex])
    block2_1 = conv_1d_layer(block2_concat, 64, 3, 1, name=name + '_block2_1',ext=iter)

    block3 = conv_1d_layer(block2_1, 128, 5, name=name + '_block3',ext=iter)
    block3_1 = conv_1d_layer(block3, 128, 3, 1, name=name + '_block3_1',ext=iter)
    block4 = conv_1d_layer(block3_1, 256, 5, name=name + '_block4',ext=iter)
    block4_1 = conv_1d_layer(block4, 256, 3, 1, name=name + '_block4_1',ext=iter)
    block5 = conv_1d_layer(block4_1, 512, 3, name=name + '_block5',ext=iter)
    block5_1 = conv_1d_layer(block5, 512, 3, 1, name=name + '_block5_1',ext=iter)

    motion_conv = conv_bn_layer(block5_1, 128, 3, 1, name=name + '_motion',ext=iter)
    fc1 = fc_bn_layer(motion_conv, 1024, name=name + '_fc1',ext=iter)
    fc2 = fc_bn_layer(fc1, 128, name=name + '_fc2',ext=iter)
    fc3 = fc_bn_layer(fc2, 7, pd.activation.Linear(), False, name=name + '_fc3',ext=iter)

    r = pd.layer.slice(input=fc3, begin=0, size=3, axis=1)
    t = pd.layer.slice(input=fc3, begin=3, size=3, axis=1)
    s = pd.layer.slice(input=fc3, begin=6, size=1, axis=1)

    outputs.append(('image_warp', image_warp))
    outputs.append(('block2_ex', block2_ex))

    for i in range(1, 6):
      exec('outputs.append((\'block' + str(i) +'\', block' + str(i) + '))')
      if i > 1:
        exec('outputs.append((\'block' + str(i) +'_1\', block' + str(i) + '_1))')

    outputs.append(('motion_conv', motion_conv))
    outputs.append(('fc1', fc1))
    outputs.append(('fc2', fc2))
    outputs.append(('fc3', fc3))

    up_block4 = deconv_bn_layer(block5_1, 256, 4, name=name + '_up_block4',ext=iter)
    up_block4 = pd.layer.concat(input=[up_block4, block4_1])
    up_block3 = deconv_bn_layer(up_block4, 128, 4, name=name + '_up_block3',ext=iter)
    up_block3 = pd.layer.concat(input=[up_block3, block3_1])
    up_block2 = deconv_bn_layer(up_block3, 64, 4, name=name + '_up_block2',ext=iter)
    up_block2 = pd.layer.concat(input=[up_block2, block2_1])
    geo_code = conv_bn_layer(up_block2, 24, name=name + '_snet_up_conv1',ext=iter)
    geo_out = conv_bn_layer(geo_code, 4, 3, 1,
                            pd.activation.Linear(), False,
                            name=name + '_snet_up_conv2',ext=iter)

    depth_noscale = pd.layer.slice(input=geo_out, begin=0, size=1, axis=1)
    normal = pd.layer.slice(input=geo_out, begin=1, size=3, axis=1)
    depth_scale = pd.layer.scaling(input=depth_noscale, weight=s)

    for i in range(2, 5)[::-1]:
      exec('outputs.append((\'up_block' + str(i) +'\', up_block' + str(i) + '))')

    eutputs.append(('geo_code', geo_code))
    outputs.append(('geo_out', geo_out))

    # outputs = OrderedDict(outputs)
    depth = pd.layer.mixed(input=[pd.layer.identity_projection(
                                  input=depth_scale)],
                           act=pd.activation.Inv())

    if not (name == 'iter_depth'):
      outputs = OrderedDict([('rotation', r),
                             ('translation', t),
                             ('scale', s),
                             ('depth_inv', depth_scale),
                             ('depth', depth),
                             ('normal', normal)])
    else:
      outputs = OrderedDict([('rotation', r),
                             ('translation', t),
                             ('scale', s),
                             ('depth_inv', depth_scale),
                             ('depth', depth),
                             ('normal', normal)])
    return outputs


def segment_block(inputs, params, name, ext_inputs):
  pass


def refine_block(inputs, params, name, ext_inputs):

    outputs = []
    depth_inv = pd.layer.bilinear_interp(input=ext_inputs['depth_inv'],
                                     out_size_x=params['size'][1],
                                     out_size_y=params['size'][0])

    image = pd.layer.concat(input=[inputs['image1'], depth_inv])
    conv0 = conv_bn_layer(image, 32, name=name + '_conv0')
    conv1 = conv_bn_layer(conv0, 64, stride=2, name=name + '_conv1')
    conv1_1 = conv_bn_layer(conv1, 64, name=name + '_conv1_1')
    conv2 = conv_bn_layer(conv1_1, 128, stride=2, name=name + '_conv2')
    conv2_1 = conv_bn_layer(conv2, 128, name=name + '_conv2_1')

    for i in range(0, 3):
      exec('outputs.append((\'conv' + str(i) +'\', conv' + str(i) + '))')
      if i > 0:
        exec('outputs.append((\'conv' + str(i) +'_1\', conv' + str(i) + '_1))')

    up_conv1 = deconv_bn_layer(conv2_1, 64, name=name + '_up_conv1')
    up_conv0 = pd.layer.concat(input=[up_conv1, conv1_1])
    up_conv0 = deconv_bn_layer(up_conv0, 32, name=name + '_up_conv0')
    up_conv0 = pd.layer.concat(input=[up_conv0, conv0])
    depth_feat = conv_bn_layer(up_conv0, 16, name=name + '_snet_conv1')
    depth_inv = conv_bn_layer(depth_feat, 1,
                          act=pd.activation.Linear(),
                          name=name + '_snet_conv2')

    depth = pd.layer.mixed(input=[pd.layer.identity_projection(
                                  input=depth_inv)],
                           act=pd.activation.Inv())

    outputs = OrderedDict(outputs)
    outputs.update(OrderedDict([('depth_0_inv',depth_inv),
                                ('depth_0', depth)]))

    return outputs


def bootstrap_net(inputs, params):
    outputs = flow_block(inputs, params, name='boost_flow')
    outputs_g = outputs
    outputs = depth_block(inputs, outputs, params, name='boost_depth')
    outputs_g.update(outputs)
    return outputs_g


def iterative_net(inputs, params):
    outputs = flow_block(inputs, params, name='iter_flow',
              ext_inputs=inputs)
    outputs_g = outputs
    outputs = depth_block(inputs, outputs, params, name='iter_depth',
              ext_inputs=inputs)
    outputs_g.update(outputs)
    return outputs_g


def refine_net(inputs, params):
    outputs = refine_block(inputs, params, name='refine_depth',
              ext_inputs=inputs)
    out_field = 'depth_0'
    outputs_g = {out_field:outputs[out_field]}
    return outputs_g


def get_demon_outputs(inputs, params, ext_inputs=None):
    """Majorly used for getting different level output.
    """
    outputs_g = {}
    if params['stage'] >= 1:
        outputs = flow_block(inputs, params, name='boost_flow')
        out_field = 'flow'
        outputs_g.update({'flow': outputs['flow']})

    if params['stage'] >= 2:
        outputs = depth_block(inputs, outputs, params, name='boost_depth')
        out_field = 'depth'
        outputs_g.update(outputs)

    if params['stage'] >= 3:
        outputs = flow_block(inputs, params, name='iter_flow',
                ext_inputs=outputs_g if ext_inputs is None else ext_inputs)
        out_field = 'flow'
        outputs_g.update({'flow': outputs['flow']})

    if params['stage'] >= 4:
        # rotation and translation need to be got from boost
        outputs = depth_block(inputs, outputs, params, name='iter_depth',
                ext_inputs=outputs_g if ext_inputs is None else ext_inputs)
        out_field = 'depth'
        outputs_g.update(outputs)

    if params['stage'] == 5:
        outputs = refine_block(inputs, params, name='refine_depth',
                  ext_inputs=outputs_g if ext_inputs is None else ext_inputs)
        out_field = 'depth_0'
        outputs_g.update({out_field:outputs[out_field]})

    return outputs_g, out_field
