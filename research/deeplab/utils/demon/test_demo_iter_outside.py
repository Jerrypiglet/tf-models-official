import gflags
import sys

paddle_root = '/home/peng/Paddle/'
sys.path.insert(0, paddle_root + 'python')
sys.path.insert(0, paddle_root + 'paddle/')
sys.path.insert(1, "./")

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

gflags.DEFINE_string('model', './output/tf_model_2.tar.gz',\
                     'The trained model for testing')
gflags.DEFINE_integer('gpu_id', 1, 'Gpu id used')
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

    # Add neural network config
    outputs_bs = d_net.bootstrap_net(inputs, params)
    outputs_it = d_net.iterative_net(inputs, params)
    outputs_re = d_net.refine_net(inputs, params)
    out_fields = ['flow', 'depth_inv', 'normal', 'rotation', 'translation']
    my_g_layer_map = {}
    parameters_bs, topo_bs = paddle.parameters.create(
      [outputs_bs[x] for x in out_fields])
    my_g_layer_map.update(cp.g_layer_map)
    parameters_it, topo_it = paddle.parameters.create(
      [outputs_it[x] for x in out_fields])
    my_g_layer_map.update(cp.g_layer_map)
    parameters_re, topo_re = paddle.parameters.create(
      outputs_re['depth_0'])
    my_g_layer_map.update(cp.g_layer_map)

    print('load parameters')
    with gzip.open(FLAGS.model, 'r') as f:
        parameters_init = paddle.parameters.Parameters.from_tar(f)

    for name in parameters_bs.names():
        parameters_bs.set(name, parameters_init.get(name))
    for name in parameters_it.names():
        parameters_it.set(name, parameters_init.get(name))
    for name in parameters_re.names():
        parameters_re.set(name, parameters_init.get(name))

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

            test_data_bs = [(image1_new, image2_new)]
            feeding_bs = {'image1': 0,
                          'image2': 1}
            flow, depth_inv, normal, rotation, translation = paddle.infer(
                                    output=topo_bs,
                                    parameters=parameters_bs,
                                    input=test_data_bs,
                                    feeding=feeding_bs);

            for i in range(3):
              test_data_it = [(image1_new, image2_new, intrinsic,
                               rotation, translation, depth_inv, normal)]
              feeding_it = {'image1': 0, 'image2': 1, 'intrinsic': 2,
                            'rotation': 3, 'translation': 4, 'depth_inv': 5,
                            'normal': 6}
              flow, depth_inv, normal, rotation, translation = paddle.infer(
                                      output=topo_it,
                                      parameters=parameters_it,
                                      input=test_data_it,
                                      feeding=feeding_it);

            test_data_re = [(image1_new, image2_new, depth_inv)]
            feeding_re = {'image1': 0, 'image2': 1, 'depth_inv': 2}
            depth = paddle.infer(output=topo_re,
                                 parameters=parameters_re,
                                 input=test_data_re,
                                 feeding=feeding_re);

            layer_names = [outputs_it['flow'].name,
                           outputs_it['normal'].name,
                           outputs_re['depth_0'].name]
            height_list = [my_g_layer_map[x].height for x in layer_names]
            width_list = [my_g_layer_map[x].width for x in layer_names]

            flow, normal, depth = vec2img(inputs=[flow, normal, depth],
                               height=height_list,
                               width=width_list)

            # visualize depth in 3D
            # image1_down = cv2.resize(image1,
            #     (depth.shape[1], depth.shape[0]))

            # visualize_prediction(
            #     depth=depth,
            #     image=np.uint8(image1_down.transpose([2, 0, 1])),
            #     rotation=rotation,
            #     translation=translation)
            with open('./test/depth_gt.npy', 'wb') as f:
                np.save(f, depth1_gt)

            with open('./test/depth_res.npy', 'wb') as f:
                np.save(f, depth)

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
