import numpy as np
import gflags

import sys
paddle_root = '/home/peng/Paddle/'
sys.path.insert(0, paddle_root + 'python')
sys.path.insert(0, paddle_root + 'paddle/')
sys.path.insert(1, "./")

import cv2
import gzip
import numpy as np
import paddle.trainer.config_parser as cp
import paddle.v2 as paddle
import pdb

import data.sun3d as sun3d
import utils.utils as uts
import layers.cost_layers as cost_layers

import network.demon_net as d_net

from paddle.utils import preprocess_util
from collections import OrderedDict

np.set_printoptions(precision=20)

gflags.DEFINE_string('model', 'tf_model_2.tar.gz',\
                     'Learning type of loss for model')
gflags.DEFINE_integer('gpu_id', 0, 'Gpu id used in the training')
FLAGS = gflags.FLAGS


def get_conv_name_matching_refine():
    name_match = {}
    for i in range(1, 6):
        name_match['block' + str(i)] = 'conv' + str(i)
        if i > 1:
            name_match['block' + str(i) + '_1'] = 'conv' + str(i) + '_1'

    for i in range(2, 5)[::-1]:
        name_match['up_block' + str(i)] = 'refine' + str(i)

    name_match['depth_0'] = 'predict_depth0'
    return name_match


def get_conv_name_matching_flow(is_iter=False):
    name_match = {}

    if is_iter:
        name_match['depth'] = 'predict_depth2'
        name_match['flow_trans'] = 'flow_trans'
        name_match['conv2_ext'] = 'block2_ext'

    for i in range(1, 6):
        name_match['block' + str(i)] = 'conv' + str(i)
        if i > 1:
            name_match['block' + str(i) + '_1'] = 'conv' + str(i) + '_1'

    name_match['code_flow'] = 'code_flow'
    name_match['flow_out_low'] = 'predict_flow5'
    name_match['up_flow'] = 'upsample_flow5to4'

    for i in range(2, 5)[::-1]:
        name_match['up_block' + str(i)] = 'refine' + str(i)

    name_match['flow_out'] = 'predict_flowconf2'

    return name_match


def get_conv_name_matching_depth(is_iter=False):
    name_match = {}
    if is_iter:
        name_match['depth_trans'] = 'depth_trans'

    for i in range(1, 6):
        name_match['block' + str(i)] = 'conv' + str(i)
        if i > 1:
            name_match['block' + str(i) + '_1'] = 'conv' + str(i) + '_1'

    name_match['image_warp'] = 'image_warp'
    name_match['block2_ex'] = 'conv2_ex'
    name_match['motion_conv'] = 'motion_conv'
    for i in range(1, 4):
        name_match['fc' + str(i)] = 'fc' + str(i)

    for i in range(2, 5)[::-1]:
        name_match['up_block' + str(i)] = 'refine' + str(i)

    name_match['geo_code'] = 'geo_code'
    name_match['geo_out'] = 'geo_out'
    name_match['depth_noscale'] = 'depth_noscale'
    name_match['scale'] = 'predict_scale'

    return name_match


def get_name_matching(stage, **kwargs):
    name_match_dict = {1: get_conv_name_matching_flow,
                       2: get_conv_name_matching_depth,
                       3: get_conv_name_matching_flow,
                       4: get_conv_name_matching_depth,
                       5: get_conv_name_matching_refine}
    name_match = name_match_dict[stage](**kwargs)
    return name_match


def vec2img(inputs, height, width):
    if not isinstance(inputs, list):
        inputs = [inputs]
        height = [height]
        width = [width]

    for i in range(len(inputs)):
        inputs[i] = inputs[i].reshape((-1, height[i], width[i]))
        inputs[i] = inputs[i].transpose((1, 2, 0))
        inputs[i] = inputs[i].squeeze()
        print inputs[i].shape

    return inputs if len(inputs) > 1 else inputs[0]


def load_tf_boost_results(folder, name_dic, stage):
    results = []
    for name in name_dic.keys():
        if name in ['flow', 'depth', 'normal']:
            cur_stage = stage-1
        elif name in ['depth_trans', 'image_warp']:
            cur_stage = stage
        else:
            cur_stage = stage-2

        with open(folder + str(cur_stage) + '_' +\
         name_dic[name] + '.pkl', 'rb') as f:
            tmp = np.load(f)
            tmp = tmp.squeeze()

        print "load name {}, shape {}".format(name, tmp.shape)
        if len(tmp.shape) == 3:
            tmp = tmp.transpose((2, 0, 1))
        results.append(tmp.flatten())

    return results


def check_diff():

    # PaddlePaddle init
    paddle.init(use_gpu=True, gpu_id=FLAGS.gpu_id)
    # paddle.init(use_gpu=False)

    # setting parameters
    params = sun3d.set_params('sun3d')
    params['stage'] = 5
    layout = [2, 3]
    cur_level = 0
    inputs = d_net.get_demon_inputs(params)


    # define several external input here to avoid implementation difference
    inputs.update(d_net.get_cnn_input("image2_down", params['size_stage'][1], 3))
    inputs.update(d_net.get_cnn_input("image_warp", params['size_stage'][1], 3))
    inputs.update(d_net.get_cnn_input("depth_trans", params['size_stage'][1], 1))
    inputs.update(d_net.get_cnn_input("flow", params['size_stage'][1], 2))

    # Add neural network config
    outputs, out_filed = d_net.get_demon_outputs(inputs, params, ext_inputs=inputs)
    print('load parameters')
    with gzip.open('./output/' + FLAGS.model, 'r') as f:
        parameters_init = paddle.parameters.Parameters.from_tar(f)

    # print parameters_init.names()
    parameters = paddle.parameters.create(outputs[out_filed])
    for name in parameters.names():
        # print "setting parameter {}".format(name)
        parameters.set(name, parameters_init.get(name))

    # load the input from saved example
    res_folder = 'output/example_output/'
    with open(res_folder + 'img_pair', 'rb') as f:
        tf_pair = np.load(f)
        tf_pair = tf_pair.squeeze()
    with open(res_folder + 'image2_down', 'rb') as f:
        image2_down = np.load(f)
        image2_down = image2_down.squeeze()
    intrinsic = np.array([0.89115971, 1.18821287, 0.5, 0.5])

    # load some extra inputs
    names = ['flow', 'depth', 'normal', 'rotation', 'translation']
    tf_names = ['predict_flow2',
                'predict_depth2',
                'predict_normal2',
                'predict_rotation',
                'predict_translation']
    start_id = range(4, 4 + len(names))
    input_name_match = dict(zip(names, tf_names))
    results_names = dict(zip(names, start_id))
    boost_results = load_tf_boost_results(res_folder, input_name_match,
                                          params['stage'])

    test_data = [tf_pair[:3, :, :].flatten(), tf_pair[3:, :, :].flatten(),
                 image2_down.flatten(), intrinsic]
    test_data = [tuple(test_data + boost_results)]
    feeding = {'image1': 0, 'image2': 1, 'image2_down': 2, 'intrinsic': 3}
    feeding.update(results_names)

    # img_diff1 = tf_pair[:3, :, :] - image1_new.reshape((3, params['size'][0], params['size'][1]))
    # img_diff1 = img_diff1.transpose((1, 2, 0))
    # uts.plot_images({'img_diff': img_diff1}, layout=[1, 2])

    # print np.sum(np.abs(tf_pair[:3, :, :].flatten() - image1_new))
    # print np.sum(np.abs(tf_pair[3:, :, :].flatten() - image2_new))

    # return
    outputs_list = [outputs[x] for x in outputs.keys()]

    # pdb.set_trace()
    print len(test_data)
    print feeding.keys()

    conv = paddle.infer(output_layer=outputs_list,
            parameters=parameters,
            input=test_data,
            feeding=feeding)

    height_list = [cp.g_layer_map[outputs[x].name].height \
                    for x in outputs.keys()]
    width_list = [cp.g_layer_map[outputs[x].name].width \
                    for x in outputs.keys()]

    conv = vec2img(inputs=conv,
                   height=height_list,
                   width=width_list)

    blob_name_match = get_name_matching(params['stage'])

    folder = './output/example_output/'
    # for name in outputs.keys()[cur_level:]:
    ob_names = outputs.keys()[cur_level:]
    # ob_names = ['depth_trans','geo_out']
    # ob_names = ['depth_0']

    for name in ob_names:
        i = outputs.keys().index(name)

        print name, ' ', blob_name_match[name]
        tf_conv_file = folder + str(params['stage']) + '_' + \
                       blob_name_match[name] + '.pkl'
        with open(tf_conv_file, 'rb') as f:
            tf_conv = np.load(f)

        print conv[i].shape, ' ', tf_conv.shape
        diff = conv[i] - tf_conv

        if len(diff.shape) <= 1:
            print '{} and {}, {}'.format(conv[i], tf_conv, diff)
        else:
            if len(diff.shape) == 2:
                diff = diff[:, :, np.newaxis]
            vis_dict = []
            for j in range(min(diff.shape[2], layout[0]*layout[1])):
                vis_dict.append(('diff_' + str(j), diff[:, :, j]))
            vis_dict = OrderedDict(vis_dict)
            uts.plot_images(OrderedDict(vis_dict), layout=layout)

def main(argv):
    argv = FLAGS(argv)
    check_diff()


if __name__ == '__main__':
    main(sys.argv)
