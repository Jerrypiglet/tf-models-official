import gflags
import sys

paddle_root = '/home/peng/Paddle/'
sys.path.insert(0, paddle_root + 'python')
sys.path.insert(0, paddle_root + 'paddle/')
sys.path.insert(1, '/home/peng/libigl/python/')
sys.path.insert(1, "../")

import pdb
import cv2
import gzip
import numpy as np
import paddle.v2 as paddle
import paddle.trainer.config_parser as cp
import pyigl as igl

import data.sun3d as sun3d
import utils.utils as uts
import utils.utils_3d as uts_3d
from utils.vis import visualize_prediction

import network.demon_net as d_net
import network.upsample_net as u_net
from paddle.utils import preprocess_util
from collections import OrderedDict
import time


gflags.DEFINE_string('model','../output/upsampler/upsample_model_l1.tar.gz',\
                     'Learning type of loss for model')
gflags.DEFINE_integer('gpu_id', 0, 'Gpu id used in the training')
gflags.DEFINE_boolean('vis', False, 'whether to visualize results')
gflags.DEFINE_string('part', '1,1', 'first value is how many parts \
                                     second value is which part to run')
FLAGS = gflags.FLAGS



def ReScale(depth_in, handle_3d, intrinsic=None, get_image_coor=False):
    """Notice handle_3d must have the same size as depth_in
    """
    depth, y, x = uts_3d.xyz2depth(handle_3d, intrinsic,
      depth_in.shape, get_image_coor=True)

    depth_in_val = depth_in[y, x]
    depth_handle = depth[y, x]

    scale = np.median(depth_handle / depth_in_val)
    depth_in *= scale

    if get_image_coor:
        height, width = depth_in.shape
        handle_3d_new = np.zeros((x.size, 3), dtype=np.float32)
        handle_3d_new[:, 0] = (np.float32(x) / width - intrinsic[2]) * \
                               depth[y, x] / intrinsic[0]
        handle_3d_new[:, 1] = (np.float32(y) / height - intrinsic[3]) * \
                               depth[y, x] / intrinsic[1]
        handle_3d_new[:, 2] = depth[y, x]
        return depth_in, y, x, handle_3d_new
    else:
        return depth_in


def CompareIn3D(depth_in, depth_gt, intrinsic=None):
    """depth_ctrl is a reference Nx3 depth map
    """

    handle_3d = uts_3d.depth2xyz(depth_gt, intrinsic, False)
    height, width = depth_in.shape[0], depth_in.shape[1]
    depth_in, y, x, handle_3d = ReScale(depth_in, handle_3d, intrinsic,
      get_image_coor=True)

    point_3d = uts_3d.depth2xyz(depth_in, intrinsic, False)
    mesh_idx = uts_3d.grid_mesh(height, width)

    U = igl.eigen.MatrixXd(np.float64(point_3d))
    U_gt = igl.eigen.MatrixXd(np.float64(handle_3d))
    F = igl.eigen.MatrixXi(np.int32(mesh_idx))
    P = igl.eigen.MatrixXd(np.float64(handle_3d))

    viewer = igl.viewer.Viewer()
    viewer.data.set_mesh(U, F)
    viewer.data.add_points(P, igl.eigen.MatrixXd([[0, 0, 1]]))
    viewer.core.is_animating = False
    viewer.launch()


def LaplacianDeform(depth_in, handle_3d, intrinsic=None, vis=False):
    """depth_ctrl is a reference Nx3 depth map
    Args:
        depth_in: the depth predicted by network
        handle_3d: the control sparse points, in [x, y, z]
        intrinsic: the intrinsic matrix
        vis: whether visualize using igl

    Output:
        depth: the warpped depth through asap
    """

    height, width = depth_in.shape[0], depth_in.shape[1]
    depth_in, y, x, handle_3d = ReScale(depth_in, handle_3d, intrinsic,
      get_image_coor=True)

    point_3d = uts_3d.depth2xyz(depth_in, intrinsic, False)
    select_id = y * width + x
    # test_id = range(10)
    # select_id = select_id[test_id]
    one_hot = np.zeros((point_3d.shape[0]), dtype=np.int32)
    one_hot[select_id] = 1
    mesh_idx = uts_3d.grid_mesh(height, width)

    V = igl.eigen.MatrixXd(np.float64(point_3d))
    U = V
    F = igl.eigen.MatrixXi(np.int32(mesh_idx))
    S = igl.eigen.MatrixXd(np.float64(one_hot))
    b = igl.eigen.MatrixXi(np.int32(select_id))

    P_origin = igl.eigen.MatrixXd(np.float64(point_3d[select_id, :]))
    P = igl.eigen.MatrixXd(np.float64(handle_3d))

    bc = igl.eigen.MatrixXd(np.float64(handle_3d))
    arap_data = igl.ARAPData()

    # Set color based on selection
    # C = igl.eigen.MatrixXd(F.rows(), 3)
    # purple = igl.eigen.MatrixXd([[80.0 / 255.0, 64.0 / 255.0, 255.0 / 255.0]])
    # gold = igl.eigen.MatrixXd([[255.0 / 255.0, 228.0 / 255.0, 58.0 / 255.0]])

    # pdb.set_trace()
    # for f in range(0, F.rows()):
    #     if S[F[f, 0]] > 0 or S[F[f, 1]] > 0 or S[F[f, 2]] > 0:
    #         C.setRow(f, purple)
    #     else:
    #         C.setRow(f, gold)

    # # Plot the mesh with pseudocolors
    # viewer = igl.viewer.Viewer()
    # viewer.data.set_mesh(V, F)
    # viewer.data.set_colors(C)
    # viewer.core.is_animating = False
    # viewer.launch()
    if vis:
        viewer = igl.viewer.Viewer()
        viewer.data.set_mesh(U, F)
        viewer.data.add_points(P, igl.eigen.MatrixXd([[1, 0, 0]]))
        viewer.core.is_animating = False
        viewer.launch()

    # start compute deform
    arap_data.max_iter = 30
    arap_data.ym = 450
    igl.arap_precomputation(V, F, V.cols(), b, arap_data)
    igl.arap_solve(bc, arap_data, U)

    if vis:
        viewer = igl.viewer.Viewer()
        viewer.data.set_mesh(V, F)
        # viewer.data.add_points(P_origin, igl.eigen.MatrixXd([[0, 0, 1]]))
        # viewer.data.add_points(P, igl.eigen.MatrixXd([[0, 1, 0]]))
        viewer.core.is_animating = False
        viewer.launch()

    point_3d_new = np.float32(np.array(U, dtype='float64', order='C'))
    depth = uts_3d.xyz2depth(point_3d_new, intrinsic, depth_in.shape)

    mask = depth <= 0
    max_depth = np.max(depth)
    depth_inpaint = cv2.inpaint(np.uint8(depth / max_depth * 255), np.uint8(mask),
                                5, cv2.INPAINT_TELEA)
    depth[mask] = np.float32(depth_inpaint[mask]) * max_depth / 255

    return depth


class DeepUpSampler(object):
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
      self.out_fields = ['flow', 'depth_inv', 'normal', 'rotation', 'translation']
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


  def demon_net_depth(self, image1, image2):
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

      layer_names = [self.outputs_re['depth_0'].name]
      height_list = [self.my_g_layer_map[x].height for x in layer_names]
      width_list = [self.my_g_layer_map[x].width for x in layer_names]

      depth = vec2img(inputs=[depth],
                      height=height_list,
                      width=width_list)

      return depth


  def UpSample(self, depth_sparse, images):
      """depth_sparse is the ground truth control points.
      """
      # flow, normal, depth1_nn, rotation, translation = self.demon_net_depth(
      #   images[0], images[1])

      depth1_nn = self.demon_net_depth(images[0], images[1])
      depth_up = LaplacianDeform(depth1_nn, depth_sparse,
             self.params['intrinsic'])

      return depth_up


def partition(num, part, part_id):
    size = num / part
    start = (part_id - 1) * size
    return range(start, min(start + size, num))


def sequencial_upsampleing(dataset='sun3d',
                           split='train',
                           max_num=None,
                           vis=False):

    # Read image pair 1, 2, generate depth
    if dataset == 'sun3d':
        params = sun3d.set_params()
        params['demon_model'] = '../output/tf_model_full_5.tar.gz'
    else:
        print "dataset {} is not supported".format(dataset)

    deep_upsampler = DeepUpSampler(params)
    part, part_id = [int(x) for x in FLAGS.part.split(',')]
    test_ids = partition(len(params[split + '_scene']), part, part_id)
    rate = 0.05
    process_scene_names = [params[split + '_scene'][x] for x in test_ids]
    all_time = 0.
    all_count = 0.

    for scene_name in process_scene_names:
        image_list = preprocess_util.list_files(
            params['flow_path'] + scene_name + '/flow/')

        image2depth = sun3d.get_image_depth_matching(scene_name)
        image_num = len(image_list) if max_num is None \
                                    else min(len(image_list), max_num)
        image_id = range(0, len(image_list), len(image_list) / image_num)
        upsample_output_path = params['flow_path'] + scene_name + \
          '/pair_depth/' + str(rate) + '/'
        uts.mkdir_if_need(upsample_output_path)

        print "processing {} with images: {}".format(
               scene_name, len(image_id))

        image_name_list = [image_list[x] for x in image_id]
        for pair_name in image_name_list:
            pair_image_name = pair_name.split('/')[-1]
            outfile = upsample_output_path + pair_image_name[:-4] + '.npy'
            # if uts.exists(outfile):
            #   print "\t {} exists".format(pair_name)
            #   continue

            image1, image2, flow_gt, depth_gt = \
                sun3d.load_image_pair(scene_name, pair_name,
                  image2depth, False)

            print pair_name
            uts.plot_images(OrderedDict([('image', image1),
                                         ('depth_gt', depth_gt)]),
                    layout=[4,2])
            continue

            depth_gt_down = uts_3d.down_sample_depth(depth_gt,
                                             method='uniform',
                                             percent=rate,
                                             K=params['intrinsic'])

            try:
              start_time = time.time()
              print "\t upsampling {}".format(pair_name)
              depth_up = deep_upsampler.UpSample(depth_gt_down,
                                                 [image1, image2])
              np.save(outfile, depth_up)
              print "\t  time: {}".format(time.time()-start_time)

              all_time +=time.time()-start_time
              all_count += 1

            except:
              print "{} failed".format(pair_name)

            if vis:
              uts.plot_images(OrderedDict([('image', image1),
                                           ('depth_gt', depth_gt),
                                           ('depth_up', depth_up)]),
                      layout=[4,2])
    print "average run time {}\n".format(all_time / all_count)


def test_refine_net(dataset='sun3d',
                    split='train',
                    vis=False):

    paddle.init(use_gpu=True, gpu_id=FLAGS.gpu_id)
    params = sun3d.set_params()
    part, part_id = [int(x) for x in FLAGS.part.split(',')]
    test_ids = partition(len(params[split + '_scene']), part, part_id)
    rate = 0.05
    is_inverse = False
    depth_name = 'depth_inv' if is_inverse else 'depth'

    process_scene_names = [params[split + '_scene'][x] for x in test_ids]
    inputs = u_net.get_inputs(params)
    outputs = u_net.refine_net(inputs, params)
    parameters, topo = paddle.parameters.create(outputs[depth_name])
    print('load parameters {}'.format(FLAGS.model))
    with gzip.open(FLAGS.model, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    feeding = {'image1':0, 'depth':1}

    for scene_name in process_scene_names:
        id_img2depth = sun3d.get_image_depth_matching(scene_name)
        upsample_output_path = params['flow_path'] + scene_name + \
          '/pair_depth/' + str(rate) + '/'
        prefix_len = len(upsample_output_path)
        image_list = preprocess_util.list_files(upsample_output_path)

        for pair_name in image_list:
            print pair_name
            pair_image_name = pair_name.split('/')[-1]
            outfile = upsample_output_path + pair_image_name[:-4] + '.npy'
            depth_net = np.load(outfile)
            depth_net_in = depth_net.flatten()
            if is_inverse:
              depth_net_in = uts_3d.inverse_depth(depth_net)

            image_name1, _ = pair_image_name.split('_')
            image_path1 = params['data_path'] + scene_name + \
                          '/image/' + image_name1 + '.jpg'
            depth_path1 = params['data_path'] + scene_name + '/depth/' + \
                          id_img2depth[image_name1] + '.png'

            image1 = cv2.imread(image_path1)
            depth1 = uts.read_depth(depth_path1)

            image1_new = uts.transform(image1.copy(),
                                       height=params['size'][0],
                                       width=params['size'][1])
            test_data = [(image1_new, depth_net_in,)]

            print 'forward'
            depth_out = paddle.infer(output=topo,
                                 parameters=parameters,
                                 input=test_data,
                                 feeding=feeding);
            if is_inverse:
              depth_out = uts_3d.inverse_depth(depth_out)

            depth = uts.vec2img(inputs=depth_out,
                                height=params['size'][0],
                                width=params['size'][1])

            if vis:
              uts.plot_images(OrderedDict([('image', image1),
                                           ('depth1', depth1),
                                           ('depth_net', depth_net),
                                           ('depth', depth)]),
                      layout=[4,2])


def test_geowarp():
    image_path1 = '/home/peng/Data/sun3d/brown_bm_1/' + \
                  'brown_bm_1/image/0001761-000059310235.jpg'
    image1 = cv2.imread(image_path1)
    with open('../test/depth_gt.npy', 'rb') as f:
        depth_gt = np.load(f)
    with open('../test/depth_res.npy', 'rb') as f:
        depth_res = np.load(f)

    if not np.all(depth_gt.shape == depth_res.shape):
        depth_gt = cv2.resize(depth_gt, (depth_res.shape[1],
          depth_res.shape[0]), interpolation=cv2.INTER_NEAREST)

    params = sun3d.set_params()
    rate = 0.05
    height, width = depth_gt.shape[0], depth_gt.shape[1]
    depth_gt_down = uts_3d.down_sample_depth(depth_gt,
                                             method='uniform',
                                             percent=rate,
                                             K=params['intrinsic'])
    depth = uts_3d.xyz2depth(depth_gt_down,
                             params['intrinsic'],
                             depth_gt.shape)

    depth_up = LaplacianDeform(depth_res, depth_gt_down,
                               params['intrinsic'], True)

    outputs, out_field = d_net.get_demon_outputs(
        inputs, params, ext_inputs=None)
    parameters, topo = paddle.parameters.create(outputs[out_field])
    uts.plot_images(OrderedDict([('image', image1),
                                 ('depth_gt', depth_gt),
                                 ('depth_down', depth),
                                 ('depth_res', depth_res),
                                 ('mask', mask),
                                 ('depth_up', depth_up)]),
                    layout=[4,2])


if __name__ == '__main__':
    argv = FLAGS(sys.argv)
    # sequencial_upsampleing('sun3d', 'train', max_num=640, vis=FLAGS.vis)
    # sequencial_upsampleing('sun3d', 'test', max_num=128, vis=FLAGS.vis)
    test_refine_net('sun3d', 'test', vis=FLAGS.vis)

