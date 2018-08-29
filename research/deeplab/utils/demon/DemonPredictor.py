import pdb
import cv2
import gzip
import numpy as np
import paddle.v2 as paddle
import paddle.trainer.config_parser as cp

import data.sun3d as sun3d
import utils.utils as uts
import utils.utils_3d as uts_3d
from utils.vis import visualize_prediction

import network.demon_net as d_net
import network.upsample_net as u_net
from paddle.utils import preprocess_util
from collections import OrderedDict
import time


class DeMoNGeoPredictor(object):
  def __init__(self, params, is_init=False):
      # PaddlePaddle init
      if not is_init:
          print("Initialize the demon deep network\n")
          paddle.init(use_gpu=True, gpu_id=FLAGS.gpu_id)

      self.params = params
      self.inputs = d_net.get_demon_inputs(self.params)

      # Add neural network config and initialize the network
      self.outputs_bs = d_net.bootstrap_net(self.inputs, self.params)
      self.outputs_it = d_net.iterative_net(self.inputs, self.params)
      self.outputs_re = d_net.refine_net(self.inputs, self.params)
      self.out_fields = ['flow', 'depth_inv', 'normal', 'rotation',
                         'translation']
      self.my_g_layer_map = {}
      self.parameters_bs, self.topo_bs = paddle.parameters.create(
        [self.outputs_bs[x] for x in self.out_fields])
      self.my_g_layer_map.update(cp.g_layer_map)
      self.parameters_it, self.topo_it = paddle.parameters.create(
        [self.outputs_it[x] for x in self.out_fields])
      self.my_g_layer_map.update(cp.g_layer_map)
      self.parameters_re, self.topo_re = paddle.parameters.create(
        self.outputs_re['depth_0'])
      self.my_g_layer_map.update(cp.g_layer_map)

      print('load parameters')
      s_time = time.time()
      print params['demon_model']
      with gzip.open(params['demon_model'], 'r') as f:
          parameters_init = paddle.parameters.Parameters.from_tar(f)

      print "load time {}".format(time.time() - s_time)
      s_time = time.time()
      for name in self.parameters_bs.names():
          self.parameters_bs.set(name, parameters_init.get(name))

      for name in self.parameters_it.names():
          self.parameters_it.set(name, parameters_init.get(name))

      for name in self.parameters_re.names():
          self.parameters_re.set(name, parameters_init.get(name))

      self.feeding_bs = {'image1': 0, 'image2': 1}
      self.feeding_it = {'image1': 0, 'image2': 1, 'intrinsic': 2,
                         'rotation': 3, 'translation': 4, 'depth_inv': 5,
                         'normal': 6}
      self.feeding_re = {'image1': 0, 'image2': 1, 'depth_inv': 2}


  def demon_geometry(self, image1, image2):
      #transform and yield
      image1_new = uts.transform(image1.copy(),
                                 height=self.params['size'][0],
                                 width=self.params['size'][1])
      image2_new = uts.transform(image2.copy(),
                                 height=self.params['size'][0],
                                 width=self.params['size'][1])
      self.intrinsic = self.params['intrinsic']

      # down sample
      test_data_bs = [(image1_new, image2_new)]
      flow, depth_inv, normal, rotation, translation = paddle.infer(
                              output=self.topo_bs,
                              parameters=self.parameters_bs,
                              input=test_data_bs,
                              feeding=self.feeding_bs);

      for i in range(3):
        test_data_it = [(image1_new, image2_new, self.intrinsic,
                         rotation, translation, depth_inv, normal)]
        flow, depth_inv, normal, rotation, translation = paddle.infer(
                                output=self.topo_it,
                                parameters=self.parameters_it,
                                input=test_data_it,
                                feeding=self.feeding_it);

      test_data_re = [(image1_new, image2_new, depth_inv)]
      depth = paddle.infer(output=self.topo_re,
                           parameters=self.parameters_re,
                           input=test_data_re,
                           feeding=self.feeding_re);

      layer_names = [self.outputs_it['flow'].name,
                     self.outputs_it['normal'].name,
                     self.outputs_re['depth_0'].name]

      height_list = [self.my_g_layer_map[x].height for x in layer_names]
      width_list = [self.my_g_layer_map[x].width for x in layer_names]

      flow, normal, depth = uts.vec2img(inputs=[flow, normal, depth],
                      height=height_list,
                      width=width_list)

      motion = np.concatenate([rotation, translation])
      return flow, normal, depth, motion
