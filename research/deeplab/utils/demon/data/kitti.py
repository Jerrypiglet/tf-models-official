# preprocess the training images
import os
import cv2
import sys

python_version = sys.version_info.major
import json
import pdb

import numpy as np
import zpark
import utils.utils as uts
import scipy.io as io
from collections import OrderedDict

HOME='/home/peng/Data/'

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1226] = 707.0493
width_to_focal[1238] = 718.3351
width_to_focal[228] = 712.3


def disp_read(file_name):
    disp = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    disp = np.float32(disp) / 256.0
    disp[disp == 0] = -1.0

    return disp


def load_car(filename):
    car_model = io.loadmat(filename)
    model = {}
    model['vertices'] = car_model['model']['hull'][0][0][0][0][1]
    model['vertices'] = model['vertices'][:, [0, 2, 1]]

    model['faces'] = car_model['model']['hull'][0][0][0][0][0]
    model['scale'] = car_model['model']['scale'][0][0][0]

    return model


def load_intrinsic(filename, mat=False):
    lines = [line for line in open(filename, 'r')]
    K = np.array([float(x) for x in lines[1].split(' ')[1:]])
    K = np.reshape(K, [3, 4])
    if mat:
        return np.float64(K[:, :3])
    else:
        return np.float64([K[0, 0], K[1, 1],
                           K[0, 2], K[1, 2]])

def depth2disp(depth, width):

    disp = width_to_focal[width] * 0.54 / depth
    return disp


def depth_read(disp):
    if isinstance(disp, str):
        disp = disp_read(disp)

    height, width = disp.shape
    assert (width in width_to_focal.keys())

    depth = width_to_focal[width] * 0.54 / disp
    depth[depth <= 0] = -1.0

    return depth


def get_instance(semantic, instance, color):
    """ get instance masks from semantic map and instance map
    """
    def get_mask(color_label, c):
        mask = np.ones((h, w), dtype=np.bool)
        for i in range(3):
            mask = np.logical_and(mask,
                    color_label[:, :, i] == np.uint8(c[i]))
        return mask

    h, w, _ = semantic.shape
    h_i, w_i, _ = instance.shape
    assert h_i == h and w_i == w
    mask = get_mask(semantic, color)
    mask_ins = []

    if np.sum(mask) == 0:
        return mask_ins

    mask = mask.reshape((h*w))
    instance_color = np.reshape(instance, (h*w, 3))
    instance_color[np.logical_not(mask), :] = 0
    color_is = np.unique(instance_color[mask, :], axis=0)
    instance_color = np.reshape(instance_color, (h, w, 3))

    for color_i in color_is:
        if np.sum(color_i) == 0:
            continue

        cur_mask = get_mask(instance, color_i)
        if np.sum(cur_mask) <= 81:
            continue
        else:
            mask_ins.append(cur_mask)

    return mask_ins


def get_instance_masks(data_params,
                       image_name,
                       class_name='car',
                       sz=None):

    semantic = cv2.imread(data_params['label_path'] + image_name + '.png',
                          cv2.IMREAD_UNCHANGED)
    # transform to rgb
    semantic = semantic[:, :, [2, 1, 0]]

    instance = cv2.imread(data_params['label_instance_path'] + \
                image_name + '.png',
                cv2.IMREAD_UNCHANGED)
    instance = instance[:, :, [2, 1, 0]]

    if not (sz is None):
        semantic = cv2.resize(semantic, (sz[1], sz[0]),
                interpolation=cv2.INTER_NEAREST)
        instance = cv2.resize(instance, (sz[1], sz[0]),
                interpolation=cv2.INTER_NEAREST)

    mask_ins = get_instance(semantic, instance, data_params['colors'][class_name])

    return mask_ins


def set_params_disp(disp='cnn'):
    params = {}
    params['data_path'] = HOME + 'kitti/2012/displets_data/displets_data/Kitti/training/'
    params['image_path'] = params['data_path'] + 'colored_0/'
    params['gt_disp_path'] = params['data_path'] + 'gt/disp_noc/'
    params['cnn_disp_path'] = params['data_path'] + 'dispmaps/%s/disp_0/' % disp
    params['sem_label_path'] = HOME + 'kitti/2012/displets_data/displets_semantic_labels/'
    params['label_path'] = params['sem_label_path'] + 'semantic/'
    params['label_instance_path'] = params['sem_label_path'] + 'instance/'
    params['car_instance_path'] = params['sem_label_path'] + 'car_instance/'
    params['car_inst_path'] = params['sem_label_path'] + 'car_inst_denoise/'
    uts.mkdir_if_need(params['car_inst_path'])
    params['output_path'] = params['data_path'] + 'Results/'

    params['car_model_path'] = HOME + 'kitti/2012/displets_data/displets_data/models/semi_convex_hull/'
    car_ids = [1, 7, 8, 9, 10, 11, 13, 15]
    params['car_names'] = ['%02d' % x for x in car_ids]
    params['car_num'] = len(car_ids)
    params['car_inst_num'] = 652

    params['image_set'] = params['data_path'] + 'split/train.txt'
    params['calib_path'] = params['data_path'] + 'calib/'
    params['intrinsic'] = []

    color_params = zpark.gen_color_list(HOME + 'zpark/color_v2.lst')
    params['size'] = [375, 1242]
    params.update(color_params)
    params['colors'] = {'car': [64, 0, 128]}
    params['class_names'] = ['car']

    return params


def set_params(val_id=-1):
    params = {}
    params['data_path'] = HOME + 'kitti/'
    params['image_path'] = params['data_path'] + 'Images/'
    params['depth_path'] = params['data_path'] + 'Depth/'
    params['pose_path'] = params['data_path'] + 'Results/Loc/'
    params['label_path'] = params['data_path'] + 'LabelFull/'
    # full with manually labelled object
    params['label_color_path'] = params['data_path'] + 'LabelFullColor/'

    # path directly rendered
    params['label_bkg_path'] = params['data_path'] + 'LabelBkg/'
    # bkg with manually impainted building
    params['label_bkgfull_path'] = params['data_path'] + 'LabelBkgFull/'
    # bkg with single object foreground only
    params['label_bkgobj_path'] = params['data_path'] + 'LabelBkgObj/'

    shader_path = "/home/peng/test/baidu/personal-code/projector/src/"
    params['vertex'] = shader_path + "PointLabel.vertexshader"
    params['geometry'] = shader_path + "PointLabel.geometryshader"
    params['frag'] = shader_path + "PointLabel.fragmentshader"
    params['is_color_render'] = True

    # shader_path = "/home/peng/test/baidu/personal-code/projector/shaderQuad/"
    # params['vertex'] = shader_path + "PointLabel.vert";
    # params['geometry'] = shader_path + "PointLabel.geom";
    # params['frag'] = shader_path + "PointLabel.frag";
    params['cloud'] = "/home/peng/Data/zpark/BkgCloud.pcd";
    # params['cloud'] = "/home/peng/Data/zpark/cluster1686.pcd";

    params['label_obj_path'] = params['data_path'] + 'Label_object/0918_moving/label/'
    params['label_color_path_v1'] = params['data_path'] + 'SemanticLabel/'
    params['label_path_v1'] = params['data_path'] + 'Label/'

    params['output_path'] = params['data_path'] + 'Results/'
    params['train_set'] = params['data_path'] + 'split/train.txt'
    params['test_set'] = params['data_path'] + 'split/val.txt'

    # simulated seqences for testing
    params['sim_path'] = params['data_path'] + 'sim_test/'

    # height width
    params['size'] = [256, 304]
    params['in_size'] = [512, 608]
    params['out_size'] = [128, 152]
    params['size_stage'] = [[8, 9], [64, 76]]
    params['batch_size'] = 4
    scenes = os.listdir(params['image_path'])
    uts.rm_b_from_a(scenes, ['gt_video.avi'])
    params['scene_names'] = []
    params['camera'] = ['Camera_1', 'Camera_2']
    for scene in scenes:
        for camera in params['camera']:
            params['scene_names'].append(scene + '/' + camera)
