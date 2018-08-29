# preprocess the training images
import os
import scipy.io as io
import glob
import cv2
import sys

python_version = sys.version_info.major
import json
import pdb

import numpy as np
import zpark
import utils.utils_3d as uts_3d
import utils.utils as uts
from collections import OrderedDict

HOME='/home/peng/Data/'
def strs_to_mat(strs):
    assert len(strs) == 4
    mat = np.zeros((4, 4))
    for i in range(4):
        mat[i, :] = np.array([np.float32(str_f) for str_f in strs[i].split(' ')])

    return mat


def get_stereo_rect_extern(params):
    camera_names = params['intrinsic'].keys()
    # relative pose of camera 2 wrt camera 1
    R = np.array([
            [ 9.96978057e-01,  3.91718762e-02, -6.70849865e-02],
            [-3.93257593e-02,  9.99225970e-01, -9.74686202e-04],
            [ 6.69948100e-02,  3.60985263e-03,  9.97746748e-01]])
    T = np.array([-0.6213358,   0.02198739,  -0.01986043])

    cx1 = 1686.23787612802
    cy1 = 1354.98486439791
    fx1 = 2304.54786556982
    fy1 = 2305.875668062

    cameraMatrix1 = np.array(
        [
            [fx1, 0, cx1],
            [0, fy1, cy1],
            [0, 0, 1.0]
        ]
    )

    cx2 = 1713.21615190657
    cy2 = 1342.91100799715
    fx2 = 2300.39065314361
    fy2 = 2301.31478860597
    cameraMatrix2 = np.array(
        [
            [fx2, 0, cx2],
            [0, fy2, cy2],
            [0, 0, 1.0]
        ]
    )

    distCoeff = np.zeros(4)
    distCoeff1 = np.array([-0.204411752675382, 0.0054170420319132,\
                           -0.0004589159435455, -0.0001137466087539])
    distCoeff2 = np.array([-0.202726777185713, -0.0130689997807488,\
                           2.97712941953E-5, -2.80495153427E-5])

    image_size = (params['size'][1], params['size'][0])
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=cameraMatrix1,
        distCoeffs1=distCoeff,
        cameraMatrix2=cameraMatrix2,
        distCoeffs2=distCoeff,
        imageSize=image_size,
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1)

    map1x, map1y = cv2.initUndistortRectifyMap(
        cameraMatrix=cameraMatrix1,
        distCoeffs=distCoeff,
        R=R1,
        newCameraMatrix=P1,
        size=image_size,
        m1type=cv2.CV_32FC1)

    map2x, map2y = cv2.initUndistortRectifyMap(
        cameraMatrix=cameraMatrix2,
        distCoeffs=distCoeff,
        R=R2,
        newCameraMatrix=P2,
        size=image_size,
        m1type=cv2.CV_32FC1)

    res = {camera_names[0] + '_rot': R1,
           camera_names[0] + '_intr': P1,
           camera_names[0] + '_mapx': map1x,
           camera_names[0] + '_mapy': map1y,
           camera_names[1] + '_rot': R2,
           camera_names[1] + '_intr': P2,
           camera_names[1] + '_mapx': map2x,
           camera_names[1] + '_mapy': map2y}

    for name in camera_names:
        res[name + '_intr'] = uts_3d.intrinsic_mat_to_vec(res[name + '_intr'])
        res[name + '_intr'][[0, 2]] /= image_size[0]
        res[name + '_intr'][[1, 3]] /= image_size[1]
        rect_extr_mat = np.eye(4)
        rect_extr_mat[:3, :3] = res[name + '_rot']
        res[name + '_ext'] = rect_extr_mat

    return res


def rect_image(image, params, camera_name, interpolation=cv2.INTER_LINEAR):
    if not (image.shape[0] == params['size'][0]):
        sz = (params['size'][1], params['size'][0])
        image = cv2.resize(image.copy(), sz,
                interpolation=cv2.INTER_NEAREST)

    image_rect = cv2.remap(image,
                    params[camera_name + '_mapx'],
                    params[camera_name + '_mapy'],
                    interpolation)

    return image_rect


def crop_image(raw_img, crop_in):
    crop = crop_in.copy()
    if np.max(np.array(crop)) < 1.0:
        h, w = raw_img.shape[:2]
        crop[[0, 2]] *= h
        crop[[1, 3]] *= w
        crop = np.uint32(crop)

    if np.ndim(raw_img) == 2:
        cropped_img = raw_img[crop[0]:crop[2], crop[1]:crop[3]] #image_size(768, 2048)
    elif np.ndim(raw_img) == 3:
        cropped_img = raw_img[crop[0]:crop[2], crop[1]:crop[3], :] #image_size(768, 2048)
    else:
        raise ValueError('not support')

    # resized_img = cv2.resize(cropped_img, None, fx=0.5, fy=0.5,
    #         interpolation = cv2.INTER_CUBIC) #image_size(384,1024)
    return cropped_img


def load_car(filename):
    car_model = io.loadmat(filename)
    model = {}
    model['vertices'] = car_model['mesh']['vertices'][0][0] / 100.0
    model['faces'] = car_model['mesh']['faces'][0][0]
    model['vertices'][:, [0, 1]] = -1 * model['vertices'][:, [0, 1]]

    return model


def read_carpose(file_name, is_euler=False, with_box=False):
    cars = []
    lines = [line.strip() for line in open(file_name)]
    i = 0
    while i < len(lines):
        car = OrderedDict([])
        line = lines[i].strip()
        if 'Model Name :' in line:
            car['name'] = line[len('Model Name : '):]
            pose = strs_to_mat(lines[i + 2: i + 6])
            pose[:3, 3] = pose[:3, 3] / 100.0
            if is_euler:
                rot = uts_3d.rotation_matrix_to_euler_angles(
                        pose[:3, :3], check=False)
                trans = pose[:3, 3].flatten()
                pose = np.hstack([rot, trans])
            car['pose'] = pose
            i += 6
            if with_box:
                while not ('Bordering Box :' in lines[i]):
                    i += 1
                car['box'] = np.array([np.uint32(x) for x in lines[i+1].split(' ')])
            cars.append(car)
        else:
            i += 1

    return cars


def cnn_depth_read(file_name,
        focal_len=None, baseline=None,
        is_crop=False, image_size=None):

    focal_len = 1053.66826 if focal_len is None else focal_len
    baseline = 1.3/1.60758404 if baseline is None else baseline

    disp = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    if file_name[-3:] == 'png':
        disp = np.float32(disp) / 256.0
    depth = focal_len * baseline / np.float32(disp)

    if is_crop:
        depth_out = np.zeros(image_size)
    else:
        depth_out = depth

    return depth_out


def depth_read(file_name):
    depth = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    depth = np.float32(depth) / 100.0
    return depth


def read_instance(label):
    label = cv2.imread(label, cv2.IMREAD_UNCHANGED)
    return label


def load_carnames(car_model_path):
    car_model_names = []
    car_types = sorted(os.listdir(car_model_path))
    for car_type in car_types:
        car_names = [os.path.basename(x)[:-4] for x in \
                glob.glob('%s/%s/*.mat' % (car_model_path, car_type))]
        for car_name in car_names:
            car_model_names.append((car_name, car_type))
    return car_types, car_model_names


HOME='/home/peng/Data/'
def set_params_disp(disp='psm', stereo_rect=True):
    params = {}
    params['data_path'] = HOME + 'apolloscape/car_labels/'
    params['image_path'] = params['data_path'] + 'images/'
    params['depth_gt_path'] = params['data_path'] + 'depth/'
    params['depth_path_rect'] = params['data_path'] + 'stereo_depth/%s/' % disp
    params['depth_path_v2'] = params['data_path'] + 'depth_2/'
    params['car_pose_path'] = params['data_path'] + 'car_poses/'
    params['car_pose_path_new'] = params['data_path'] + 'car_poses_challenge/'
    params['car_pose_path_sim'] = params['data_path'] + 'car_poses_challenge_sim/'
    uts.mkdir_if_need(params['car_pose_path_new'])
    uts.mkdir_if_need(params['car_pose_path_sim'])

    params['car_inst_path'] = params['data_path'] + 'car_instance/'
    params['train_list'] = params['data_path'] + 'split/train.txt'
    params['val_list'] = params['data_path'] + 'split/val.txt'
    params['minival_list'] = params['data_path'] + 'split/mini_val.txt'
    params['minitrain_list'] = params['data_path'] + 'split/mini_train.txt'
    params['output_path'] = params['data_path'] + 'results/'
    params['car_inst_num'] = 10000

    params['size'] = [2710, 3384]
    params['intrinsic'] = {
            'Camera_5': np.array([2304.54786556982, 2305.875668062,
                1686.23787612802, 1354.98486439791]),
            'Camera_6': np.array([2300.39065314361, 2301.31478860597,
                1713.21615190657, 1342.91100799715])}

    # normalized intrinsic
    cam_names = params['intrinsic'].keys()
    for c_name in cam_names:
        params['intrinsic'][c_name][[0, 2]] /= params['size'][1]
        params['intrinsic'][c_name][[1, 3]] /= params['size'][0]

    vertice_num = '5000'
    params['car_model_path'] = HOME + 'car_models/%s/' % vertice_num
    params['car_model_path'] = HOME + 'car_models/5000_align/'
    params['car_model_path_pkl'] = params['data_path'] + '/3d_car_instance_sample/car_models/'
    params['car_model_path_off'] = HOME + '/car_models/%s_off/' % vertice_num
    params['car_model_path_pkl'] = HOME + '/car_models/%s_pkl/' % vertice_num
    params['save_image_path'] = params['data_path'] + '/3d_car_instance_sample/images/'

    # params['car_model_path'] = HOME + 'car_models/5000/'
    # params['car_model_path_pkl'] = HOME + 'car_models/5000_pkl/'
    uts.mkdir_if_need(params['car_model_path_pkl'])
    uts.mkdir_if_need(params['car_model_path_off'])

    # params['car_names'] = [os.path.basename(x)[:-4] for x in \
    #         glob.glob(params['car_model_path'] + '*.mat')]
    params['car_types'], params['car_names'] = load_carnames(params['car_model_path'])

    color_params = zpark.gen_color_list(HOME + 'zpark/color_v2.lst')
    params.update(color_params)

    if stereo_rect:
        # the crop with cars
        params['rect_crop'] = np.array([1432., 668., 2200., 2716.])
        params['rect_crop'][[0, 2]] /= params['size'][0]
        params['rect_crop'][[1, 3]] /= params['size'][1]

        # the crop only crop vertically
        params['rect_crop_top'] = np.array([1432., 0., 2200., 3383.])
        params['rect_crop_top'][[0, 2]] /= params['size'][0]
        params['rect_crop_top'][[1, 3]] /= params['size'][1]


        params['image_path_rect'] = params['image_path'] + 'stereo_rect/'
        uts.mkdir_if_need(params['image_path_rect'])
        params['car_inst_path_rect'] = params['car_inst_path'] + 'stereo_rect/'
        uts.mkdir_if_need(params['car_inst_path_rect'])
        rectify_params = get_stereo_rect_extern(params)
        params.update(rectify_params)

    return params



if __name__ == '__main__':
    import utils.utils as uts
    import time
    params = set_params_disp(stereo_rect=True)
    for i, name in enumerate(params['car_names']):
        print "Label(%30s, %10d, %10s, %10d)," % ("'" + name[0] + "'", i, "'" + name[1] + "'", params['car_types'].index(name[1]))

    # test_image = [line.strip() for line in open(params['train_list'])]
    # test_image_name = test_image[0][:-4]
    # camera_name = 'Camera_5'
    # s = time.time()
    # image = cv2.imread(params['image_path'] + test_image_name + '.jpg')
    # print 'reading time', time.time() - s, test_image_name
    # depth_gt = depth_read(params['depth_path_v2'] + test_image_name + '.png')
    # sz = (params['size'][1], params['size'][0])
    # depth_gt = cv2.resize(depth_gt, sz, interpolation=cv2.INTER_NEAREST)
    # mask_gt = cv2.imread(params['car_inst_path'] + test_image_name + '.png',
    #         cv2.IMREAD_UNCHANGED)
    # mask_gt_ori = cv2.resize(mask_gt, sz, interpolation=cv2.INTER_NEAREST)

    # image_rect = rect_image(image, params, camera_name)
    # depth_gt = rect_image(depth_gt, params, camera_name)
    # mask_gt = rect_image(mask_gt_ori, params, camera_name)

    # print 'processing time', time.time() - s
    # image_crop = crop_image(image_rect)
    # mask_gt = crop_image(mask_gt)
    # depth_gt = crop_image(depth_gt)

    # depth = cnn_depth_read(params['depth_path'] + test_image_name + '.jpg')
    # image_crop[:, :, 2] = 100 * mask_gt
    # image[:, :, 2] = 100 * mask_gt_ori
    # uts.plot_images({'image': image, 'image_rect': image_rect,
    #         'mask': mask_gt, 'depth_gt': depth_gt,
    #         'crop_image': image_crop, 'depth': depth}, layout=[2, 3])


