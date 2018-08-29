
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
    disp[disp == 0] = -1.0

    return disp


def depth_read(disp):
    if isinstance(disp, str):
        disp = disp_read(disp)

    depth = 1053.66826 * 0.5372 / disp
    depth[depth <= 0] = -1.0
    return depth


def depth_read_png(file_name, scale=700.0):
    depth = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    depth = np.float32(depth) / scale
    return depth


def read_instance(label):
    label = cv2.imread(label, cv2.IMREAD_UNCHANGED)
    return label


HOME='/home/peng/Data/'
def set_params():
    params = {}
    params['data_path'] = HOME + 'xroad/'
    params['image_path'] = params['data_path'] + 'test_samples/'
    params['gt_disp_path'] = params['data_path'] + 'gt/disp_noc/'
    params['sem_label_path'] = HOME + 'kitti/2012/displets_data/displets_semantic_labels/'
    params['label_path'] = params['sem_label_path'] + 'semantic/'
    params['label_instance_path'] = params['sem_label_path'] + 'instance/'
    params['car_instance_path'] = params['sem_label_path'] + 'car_instance/'
    params['image_set'] = params['data_path'] + 'split/train.txt'
    params['calib_path'] = params['data_path'] + 'calib/'

    params['intrinsic_v1'] = [1053.66826, 1053.66826,
                              518.730500, 141.9679700]
    params['size'] = [384, 1024]
    color_params = zpark.gen_color_list(HOME + 'zpark/color_v2.lst')
    params.update(color_params)

    return params
