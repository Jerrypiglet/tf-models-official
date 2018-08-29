import gflags
import sys

paddle_root = '/home/peng/Paddle/'
sys.path.insert(0, paddle_root + 'python')
sys.path.insert(0, paddle_root + 'paddle/')
sys.path.insert(1, "../")

import cv2
import gzip
import numpy as np
import paddle.v2 as paddle
import paddle.trainer.config_parser as cp

import data.sun3d as sun3d
import utils.utils as uts
from utils.vis import visualize_prediction

import layers.cost_layers as cost_layers
import network.demon_net as d_net
from paddle.utils import preprocess_util
from collections import OrderedDict

gflags.DEFINE_string('model', 'tf_model_2.tar.gz',\
                     'Learning type of loss for model')
gflags.DEFINE_integer('gpu_id', 0, 'Gpu id used in the training')
FLAGS = gflags.FLAGS


def vec2img(inputs, height, width):
    if not isinstance(inputs, list):
        inputs = [inputs]
        height = [height]
        width = [width]

    for i in range(len(inputs)):
        inputs[i] = inputs[i].reshape((-1, height[i], width[i]))
        inputs[i] = inputs[i].transpose((1, 2, 0))
        inputs[i] = inputs[i].squeeze()

    return inputs if len(inputs) > 1 else inputs[0]


def test_demo():
    # PaddlePaddle init
    paddle.init(use_gpu=True, gpu_id=FLAGS.gpu_id)
    params = sun3d.set_params()
    inputs = d_net.get_demon_inputs(params)

    params['stage'] = 2

    # Add neural network config
    outputs, out_field = d_net.get_demon_outputs(inputs, params, ext_inputs=None)
    parameters = paddle.parameters.create(outputs[out_field])

    print('load parameters')
    with gzip.open(FLAGS.model, 'r') as f:
        parameters_init = paddle.parameters.Parameters.from_tar(f)
    for name in parameters.names():
        parameters.set(name, parameters_init.get(name))

    # Read image pair 1, 2 flow
    for scene_name in params['train_scene'][1:]:
        image_list = preprocess_util.list_files(
            params['flow_path'] + scene_name + '/flow/')
        image2depth = sun3d.get_image_depth_matching(scene_name)

        for pair_name in image_list[0:2]:
            image1, image2, flow_gt, depth1_gt, normal1_gt = \
                sun3d.load_image_pair(scene_name, pair_name, image2depth)

            #transform and yield
            image1_new = uts.transform(image1.copy(),
                                       height=params['size'][0],
                                       width=params['size'][1])
            image2_new = uts.transform(image2.copy(),
                                       height=params['size'][0],
                                       width=params['size'][1])
            intrinsic = np.array([0.89115971, 1.18821287, 0.5, 0.5])

            test_data = [(image1_new, image2_new, intrinsic)]
            depth_name = 'depth' if params['stage'] < 5 else 'depth_0'
            out_fields = ['flow', depth_name, 'normal', 'rotation',
                          'translation']

            output_list = [outputs[x] for x in out_fields]
            flow, depth, normal, rotation, translation = paddle.infer(
                                    output_layer=output_list,
                                    parameters=parameters,
                                    input=test_data,
                                    feeding={'image1': 0,
                                             'image2': 1,
                                             'intrinsic': 2});

            height_list = [cp.g_layer_map[outputs[x].name].height \
                            for x in ['flow', 'depth','normal']]
            width_list = [cp.g_layer_map[outputs[x].name].width \
                            for x in ['flow', 'depth','normal']]

            flow, depth, normal = vec2img(inputs=[flow, depth, normal],
                               height=height_list,
                               width=width_list)

            # visualize depth in 3D
            image1_down = cv2.resize(image1,
                (depth.shape[1], depth.shape[0]))

            visualize_prediction(
                depth=depth,
                image=np.uint8(image1_down.transpose([2, 0, 1])),
                rotation=rotation,
                translation=translation)

            uts.plot_images(OrderedDict([('image1',image1),
                                         ('image2',image2),
                                         ('flow',flow),
                                         ('flow_gt',flow_gt),
                                         ('depth', depth),
                                         ('depth_gt', depth1_gt),
                                         ('normal', (normal + 1.0)/2.),
                                         ('normal_gt', (normal1_gt + 1.0)/2)]),
                            layout=[4,2])


def main(argv):
    argv = FLAGS(argv)
    test_demo()

if __name__ == '__main__':
    main(sys.argv)
