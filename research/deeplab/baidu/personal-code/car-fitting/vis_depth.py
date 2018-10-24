import utils.utils_3d as uts_3d
import extern.render.render as render
import utils.utils as uts

import os
import cv2
import numpy as np
import scipy.io as io
from utils.vis import visualize_depths

from collections import OrderedDict
import logging

DTYPE = np.float32
np.set_printoptions(threshold=np.nan)

import data.kitti as kitti
import data.zpark as zpark
import data.xroad as xroad
import data.apolloscape as apollo

data_libs = {}
data_libs['kitti'] = kitti
data_libs['apollo'] = apollo

import evaluation.eval_utils as eval_uts
import pdb

from collections import OrderedDict

cars = OrderedDict({})
params = kitti.set_params_disp()
for car_name in params['car_names']:
    car_file = '%s/%s.mat' % (params['car_model_path'], \
                car_name)
    cars[car_name] = kitti.load_car(car_file)


def vis_example(image_id, object_id):
    data_path = '/home/peng/Data/kitti/2012/displets_data/displets_data/Kitti/training/'
    res_path = data_path + 'displets_gt_seg/cnn/'
    disp_path = data_path + '/dispmaps/cnn/disp_0/'
    gt_path = data_path + '/gt/disp_noc/'
    file_name = '%06d_10' % image_id

    print('show results of %s\n' % file_name)
    disp_mat_file = res_path + file_name + '_%02d.mat' % object_id
    cnn_depth_file = disp_path + file_name  + '.png'
    gt_depth_file = gt_path + file_name + '.png'

    intrinsic_file = data_path + '/calib/' + file_name[:-3] + '.txt'
    lines = [line for line in open(intrinsic_file, 'r')]
    K = np.array([float(x) for x in lines[0].split(' ')[1:]])
    K = np.reshape(K, [3, 4])

    displets = io.loadmat(disp_mat_file)
    hl, wl = displets['render_depth'].shape
    mask = displets['obj_mask']

    depth_cnn = kitti.disp_read(cnn_depth_file)
    h, w = depth_cnn.shape
    depth_cnn = cv2.resize(depth_cnn, (wl, hl), interpolation=cv2.INTER_NEAREST)
    depth_cnn[mask == 0] = -1.

    depth_gt = kitti.disp_read(gt_depth_file)
    mask = cv2.resize(mask, depth_gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
    depth_gt[mask == 0] = -1.
    K = np.array([K[0, 0]/w, K[1, 1]/h, K[0, 2]/w, K[1, 2]/h])

    depths = {}
    depths['cnn_depth'] = {'depth': 100.0 / depth_cnn,
                           'color': np.array([0, 255, 255])}
    depths['render_depth'] = {'depth': 100.0 / displets['render_depth'],
                              'color': np.array([255, 0, 0])}
    depths['gt_depth'] = {'depth': 100.0 / depth_gt,
                          'color': np.array([255, 255, 0])}
    visualize_depths(depths, K)


def eval_depth(gt_depths, pred_depths,
               masks=None,
               min_depth=1e-3,
               max_depth=80,
               vis_path=None):

    num_samples = len(pred_depths)
    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)

    for i in range(num_samples):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]
        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth) \
                if masks is None else masks[i]
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
                eval_uts.compute_errors(gt_depth[mask], pred_depth[mask])

    res = [abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()]
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(*res))

    return res


def eval_instance_depth(gt_instance, pred_instance, num_samples, cap_depth=np.inf):
    """ For all image evaluate the depth precision and average IOU of each instance
        Instance ID must match between gt and prediction
    """

    num_images = len(pred_instance)

    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)
    is_eval = np.ones(num_samples, dtype=np.bool)

    IOUs = np.zeros(num_samples, np.float32)
    i = 0

    for img_id in range(num_images):
        print("eval %s" % img_id)
        depth_gt = gt_instance[img_id][1]
        instance_gt = gt_instance[img_id][0]
        depth_pred = pred_instance[img_id][1]
        instance_pred = pred_instance[img_id][0]

        ids = np.unique(instance_gt[instance_gt > 0])
        all_mask = np.logical_and(depth_gt > 0, depth_pred > 0)
        # uts.plot_images({'mask': instance_gt})

        for idx in ids:
            gt_seg = instance_gt == idx
            pred_seg = instance_pred == idx
            IOUs[i] = eval_uts.IOU(gt_seg, pred_seg)
            # uts.plot_images({'gt': gt_seg, 'pred': pred_seg})
            if IOUs[i] == 0:
                continue

            mask = np.logical_and(
                    all_mask, np.logical_and(gt_seg, pred_seg))
            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
                    eval_uts.compute_errors(depth_gt[mask], depth_pred[mask])

            if depth_gt[mask].mean() > cap_depth and np.sum(mask) > 0:
                is_eval[i] = False
            print depth_gt[mask].mean(), is_eval[i], IOUs[i]

            i = i + 1

    num_samples = np.float32(num_samples)
    cond1 = np.logical_and(IOUs > 0.5, a1 > 0.6)
    cond2 = np.logical_and(IOUs > 0.7, a1 > 0.8)
    cond3 = np.logical_and(IOUs > 0.85, a1 > 0.9)

    delta1 = np.sum(cond1) / num_samples
    delta2 = np.sum(cond2) / num_samples
    delta3 = np.sum(cond3) / num_samples

    valid = np.logical_and(~np.isnan(abs_rel), is_eval)

    res = [abs_rel[valid].mean(), sq_rel[valid].mean(), rms[valid].mean(), \
            log_rms[valid].mean(), a1[valid].mean(), a2[valid].mean(), a3[valid].mean()]
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(*res))

    print("{:>10}, {:>10}, {:>10}, {:>10}".format('mean_IOU', 'rc % 0.5', '% iou > 0.75', '% iou > 0.9'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}".format(IOUs.mean(), delta1, delta2, delta3))


def eval_results_xroad():

    params = xroad.set_params()
    image_list = ['%03d' % i for i in range(100)]
    res_path = params['data_path'] + 'displets_gt_seg_v2/'
    cnn_disp_path = params['data_path'] + '/test_samples_v2/'
    gt_path = params['gt_disp_path']

    depth_name = 'render_disp'
    mask_name = 'obj_mask'
    gt_car_depths = []
    pred_car_depths = []
    masks = []

    print("load dataset")
    num_samples = 0
    # generate results pairs (instance, depth)
    for image_name in image_list:
        disp_mat_file = res_path + image_name + '_car.mat'
        if not os.path.exists(disp_mat_file):
            continue

        displets = io.loadmat(disp_mat_file)
        h_pred, w_pred = displets[depth_name].shape

        render_mask = displets[mask_name]
        depth_car = xroad.depth_read(displets[depth_name])

        # because we don't have gt depth, use render depth for gt
        depth_gt_bkg = xroad.depth_read_png(
                params['image_path'] + image_name + '_masked_depth.png', 200.0)
        depth_gt = depth_car + depth_gt_bkg
        h, w = h_pred, w_pred

        instance_gt = cv2.resize(displets['mask'], (w_pred, h_pred),
                                 interpolation=cv2.INTER_NEAREST)
        inst_id = np.unique(instance_gt[:])
        num_samples += len(inst_id[inst_id > 0])

        pred_car_depths.append((render_mask, depth_car))
        gt_car_depths.append((instance_gt, depth_gt))

        mask = np.logical_and(depth_car > 0, depth_gt > 0)
        masks.append(mask)

        # show gt depth gt instance
        # pdb.set_trace()
        if True:
            image_file = params['image_path'] + image_name + '_rgb.jpg'
            image = cv2.imread(image_file)
            depth_gt_masked = depth_gt.copy()
            depth_gt_masked[instance_gt <= 0] = -1
            image = image[:, :, ::-1]

            cnn_depth_file = cnn_disp_path + image_name + '_depth.png'
            depth_cnn = xroad.depth_read_png(cnn_depth_file)
            depth_cnn[render_mask == 0] = -1.

            label_c = uts.label2color(render_mask, params['color_map_list'],
                    [255,255,255])
            alpha = 0.7
            image = np.uint8(alpha * image + (1-alpha) * label_c);

            uts.plot_images(OrderedDict({'image': image,
                                         'pred_instance': render_mask,
                                         'pred_depth': depth_car,
                                         'gt_instance': instance_gt,
                                         'gt_depth': depth_gt_masked,
                                         'depth_cnn':depth_cnn}), layout=[2, 3])

            K = params['intrinsic_v1']
            K = np.array([K[0]/w, K[1]/h, K[2]/w, K[3]/h])
            depths = {}
            depths['cnn_depth'] = {'depth': depth_cnn,
                                   'color': np.array([0, 255, 255])}
            depths['render_depth'] = {'depth': depth_car,
                                      'color': np.array([255, 0, 0])}
            depths['gt_depth'] = {'depth': depth_gt_masked,
                                  'color': np.array([255, 255, 0])}

            visualize_depths(depths, K)

    # eval_depth(gt_car_depths, pred_car_depths, masks)
    print num_samples
    eval_instance_depth(gt_car_depths, pred_car_depths, num_samples)


def plot_3d_boxes(params, image, intrinsic, poses, model_ids, cars):

    boxes_3d = []
    # image_size = image.shape[:2]

    # adjust to runing size
    image_size = [188, 624]
    # intrinsic[:2, :] *= (np.float32(188)/np.float32(image.shape[0]))
    # image = cv2.resize(image, tuple(image_size[::-1]))

    # print image_size
    intr_vec = np.float64(uts_3d.intrinsic_mat_to_vec(intrinsic))
    for pose, model_id in zip(poses, model_ids):
        car = cars[car_name]
        pose_plot = pose.copy()
        pose_plot[4] = -1 * (pose_plot[4] + 1.6)
        # print pose_plot
        vert = uts_3d.project(pose_plot, car['scale'], car['vertices'])
        depth, mask = render.renderMesh_np(
                np.float64(vert),
                np.float64(car['faces']),
                intr_vec/2, image_size[0], image_size[1])
        depth[mask == 0] = -1.0

        vert[:, 1] = -1 * vert[:, 1]
        boxes_3d.append(np.vstack([np.min(vert, axis=0), \
                                   np.max(vert, axis=0)]))

    image = uts_3d.draw_3dboxes(image, boxes_3d, intrinsic)
    # sz = image.shape[:2][::-1]
    # mask = cv2.resize(mask, sz)
    # mask = np.concatenate([mask[:, :, None] for i in range(3)], axis=2) * 255
    # mask = np.uint8(mask * 0.7 + np.float32(image) * 0.3)
    # uts.plot_images(OrderedDict({'image': image,
    #     'depth': depth, 'mask': mask}))

    return image


def visualize_res(params,
                  image_name,
                  render_mask,
                  depth_car,
                  poses,
                  model_ids,
                  depth_gt,
                  instance_gt,
                  w, h, cnn_disp_path):

    intrinsic_file = params['calib_path'] + image_name[:-3] + '.txt'
    lines = [line for line in open(intrinsic_file, 'r')]
    K = np.array([float(x) for x in lines[0].split(' ')[1:]])
    K = np.reshape(K, [3, 4])

    image_file = params['image_path'] + image_name + '.png'
    image = cv2.imread(image_file)
    image = image[:, :, ::-1]

    image = plot_3d_boxes(params, image, K[:, :3], poses, model_ids, cars)
    # uts.plot_images({'image': image})

    label_c = uts.label2color(
            render_mask, params['color_map_list'],
            [255,255,255])
    alpha = 0.7
    image = np.uint8(alpha * image + (1 - alpha) * label_c);

    depth_gt_masked = depth_gt.copy()
    depth_gt_masked[instance_gt <= 0] = -1
    depth_car[render_mask <=0 ] = -1

    cnn_depth_file = cnn_disp_path + image_name + '.png'
    depth_cnn = kitti.depth_read(cnn_depth_file)
    depth_cnn[render_mask <= 0] = -1.
    uts.plot_images(OrderedDict(
                   {'image': image,
                    'pred_instance': render_mask,
                    'pred_depth': depth_car,
                    'gt_instance': instance_gt,
                    'gt_depth': depth_gt_masked,
                    'cnn_depth': depth_cnn}), layout=[2, 3])

    K = np.array([K[0, 0]/w, K[1, 1]/h, K[0, 2]/w, K[1, 2]/h])
    depths = {}
    depths['cnn_depth'] = {'depth': depth_cnn,
                           'color': np.array([0, 255, 255])}
    depths['render_depth'] = {'depth': depth_car,
                              'color': np.array([255, 0, 0])}
    depths['gt_depth'] = {'depth': depth_gt_masked,
                          'color': np.array([255, 255, 0])}
    visualize_depths(depths, K)


def eval_results(data='kitti'):

    # params = eval('data.' + dataset + '.set_params()')
    disp = 'psm'
    params = kitti.set_params_disp()
    image_list = ['%06d_10' % i for i in range(1, 194)]
    with open(params['data_path'] + 'file_list.txt', 'a') as f:
        for image_name in image_list:
            f.write(image_name + '\n')

    res_path = params['data_path'] + 'displets_gt_seg/%s/' % disp
    cnn_disp_path = params['data_path'] + '/dispmaps/%s/disp_0/' % disp
    gt_path = params['gt_disp_path']

    depth_name = 'render_disp'
    mask_name = 'obj_mask'
    gt_car_depths = []
    pred_car_depths = []
    masks = []
    cap_depth = 40.0

    print("load dataset")
    num_samples = 0

    # generate results pairs (instance, depth)
    for image_name in image_list:
        disp_mat_file = res_path + image_name + '_car.mat'
        if not os.path.exists(disp_mat_file):
            continue

        gt_depth_file = gt_path + image_name + '.png'
        gt_instance_file = params['car_instance_path'] + image_name + '.png'

        instance_gt = cv2.imread(gt_instance_file, cv2.IMREAD_UNCHANGED)
        inst_id = np.unique(instance_gt[:])
        num_samples += len(inst_id[inst_id > 0])

        depth_gt = kitti.depth_read(gt_depth_file)
        h, w = depth_gt.shape
        gt_car_depths.append((instance_gt, depth_gt))

        displets = io.loadmat(disp_mat_file)
        h_pred, w_pred = displets[depth_name].shape

        if h_pred != h:
            displets[mask_name] = cv2.resize(displets[mask_name],
                    (w, h), interpolation=cv2.INTER_NEAREST)
            displets[depth_name] = cv2.resize(displets[depth_name],
                    (w, h), interpolation=cv2.INTER_NEAREST)
            instance_gt = cv2.resize(displets['mask'], (w, h),
                    interpolation=cv2.INTER_NEAREST)

        render_mask = displets['obj_mask']
        depth_car = kitti.depth_read(displets[depth_name])

        poses = displets['poses']
        poses_np = np.zeros((poses.shape[0], 6))
        poses_np[:, [1, 3, 4, 5]] = poses
        model_ids = displets['car_ids']

        pred_car_depths.append((render_mask, depth_car))
        mask = np.logical_and(depth_car > 0, depth_gt > 0)
        masks.append(mask)

        # show gt depth gt instance
        if False:
            visualize_res(params, image_name,
                          render_mask, depth_car, poses_np, model_ids,
                          depth_gt, instance_gt, w, h, cnn_disp_path)

    # eval_depth(gt_car_depths, pred_car_depths, masks)
    print num_samples, cap_depth
    eval_instance_depth(gt_car_depths, pred_car_depths, num_samples, cap_depth)


def vis_depth(image_id, method='psm', folder='testing'):

    data_path = '/home/peng/Data/kitti/2012/displets_data/displets_data/Kitti/%s/' % folder
    file_name = '%06d_10' % image_id

    if not isinstance(method, list):
        method = [method]

    colors = [[255, 0, 0], [0, 255, 0], [225, 255, 0], [0, 255, 255]]
    print('show results of %s' % file_name)

    intrinsic_file = data_path + '/calib/' + file_name[:-3] + '.txt'
    lines = [line for line in open(intrinsic_file, 'r')]
    K = np.array([float(x) for x in lines[0].split(' ')[1:]])
    K = np.reshape(K, [3, 4])

    depths = {}
    for i, m in enumerate(method):
        disp_path = data_path + '/dispmaps/%s/disp_0/' % m
        cnn_depth_file = disp_path + file_name  + '.png'
        depth_cnn = kitti.disp_read(cnn_depth_file)
        h, w = depth_cnn.shape
        if i == 0:
            K = np.array([K[0, 0]/w, K[1, 1]/h, K[0, 2]/w, K[1, 2]/h])

        depths[m] = {'depth': 100.0 / depth_cnn,
                     'color': np.array(colors[i])}

    visualize_depths(depths, K)


def vis_xroad_depth():
    data_folder = '/home/peng/Data/xroad/test_samples/'
    file_name = data_folder + '001_depth.png'
    mask_file = data_folder + '001_instanceIds.png'

    depth = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    depth = np.float32(depth)/200.0
    h, w = depth.shape

    segment = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
    segment = np.logical_and(segment/1000 == 33, segment % 1000 == 1)

    uts.plot_images({'segment':segment, 'depth':depth})

    depth[segment == 0] = -1
    print np.mean(depth[segment])

    K = [1053.7/w, 1053.7/h, 518.7/w, 149.6/h]
    # np.array([K[0, 0]/w, K[1, 1]/h, K[0, 2]/w, K[1, 2]/h])

    depths = {}
    depths['cnn_depth'] = {'depth': depth,
                           'color': np.array([0, 255, 255])}
    depths['rendered_depth'] = {'depth': depth_car,
            'color': np.array([255, 255, 0])}
    visualize_depths(depths, K)


def vis_apollo_depth(image_name):

    def post_processing(depth, data_params):
        s = 0.5
        image_size = np.uint32(
                [data_params['size'][0] * s, data_params['size'][1] * s])
        depth = uts.padding_image(depth, data_params['rect_crop_top'],
                    image_size.tolist(), pad_val=-1.)
        depth[depth > 80] = -1.0
        return depth


    dataname = 'apollo'
    data_params = data_libs['apollo'].set_params_disp()
    depth_path_new = '/media/peng/DATA/Data/apolloscape/car_labels/stereo_rgbd_top_crop_png_746/%s.png' % image_name
    depth_path_old = '/media/peng/DATA/Data/apolloscape/car_labels/stereo_rgb_top_crop_png/%s.png' % image_name
    depth_gt_file  = data_params['depth_path_v2'] + image_name + '.png'

    s = 0.5

    h, w = [int(data_params['size'][0] * s), int(data_params['size'][1] * s)]
    depth_gt = data_libs[dataname].depth_read(depth_gt_file)
    depth_gt = cv2.resize(depth_gt, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = depth_gt > 0

    depth1 = data_libs[dataname].cnn_depth_read_v2(depth_path_new)
    depth1 = post_processing(depth1, data_params) *  mask

    depth = data_libs[dataname].cnn_depth_read(depth_path_old)
    depth = post_processing(depth, data_params) * mask

    # uts.plot_images({'depth_psm': depth, 'depth_fine': depth1, 'depth_gt': depth_gt})

    # depth1 = cv2.resize(depth1,
    # depth2 = cv2.imread(depth_path_old, cv2.IMREAD_UNCHANGED)
    # depth1 = pad(depth1)
    # depth2 = pad(depth2)

    depths = {}
    depths['depth_gt'] = {'depth': depth_gt,
            'color': np.array([0, 255, 0])}
    depths['cnn_depth_new'] = {'depth': depth1,
            'color': np.array([255, 0, 0])}
    depths['cnn_depth_old'] = {'depth': depth,
            'color': np.array([0, 255, 255])}
    # depths['depth2'] = {'depth': depth2,
    #         'color': np.array([255, 255, 0]}

    K = data_params['intrinsic']['Camera_5']
    visualize_depths(depths, K)


def vis_apollo_render_depth(self):
    # sample an image
    if self.counter == self.image_num:
        self.counter = 0
        # logging.info('inst number %d' % self.inst_counter_g)
        self.inst_counter_g = 0
        self.list_order = np.random.permutation(self.image_num)

    idx = self.list_order[self.counter]
    image_name = self.image_list[idx]
    self.image_name = image_name[:-4]

    image_path = self.data_params['image_path']
    self.image = cv2.imread(image_path \
              + self.image_name + '.jpg', cv2.IMREAD_UNCHANGED)
    for name in self.data_params['intrinsic'].keys():
        if name in image_name:
            cam_name = name
            break

    self.intrinsic = uts_3d.intrinsic_vec_to_mat(
                self.data_params['intrinsic'][cam_name],
                self.image_size)
    self.focal_len = self.intrinsic[0, 0]

    gen_depth = True
    mask_path = self.data_params['car_inst_path']
    depth_path = self.data_params['depth_path_v2']
    depth_file = depth_path + self.image_name + '.png'
    mask_file = mask_path + self.image_name + '.png'
    if os.path.exists(depth_file):
        self.depth = data_libs[self.d_name].depth_read(depth_file)
        self.masks = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        gen_depth = False

    # sample an instance
    self.inst_counter = 0
    if self.scale != 1.0:
        hs, ws = self.image_size
        h, w = self.image_size_ori
        self.image = cv2.resize(self.image, (ws, hs))
        if not gen_depth:
            self.depth = cv2.resize(self.depth, (ws, hs),
                    interpolation=cv2.INTER_NEAREST)
            self.masks = cv2.resize(self.masks, (ws, hs),
                    interpolation=cv2.INTER_NEAREST)

    # generate depth & mask from the given poses
    pose_file = self.data_params['car_pose_path'] + \
            self.image_name + '.poses'
    if not os.path.exists(pose_file):
        return False
    self.car_poses = data_libs[self.d_name].read_carpose(
            pose_file, is_euler=True)

    if gen_depth:
        self.masks = np.zeros(self.depth.shape)
        self.depth = 10000. * np.ones(self.image_size)

        for inst_id in range(len(self.car_poses)):
            car_model = self.car_poses[inst_id]
            # if not (car_model['name'] in self.car_model):
            #     car_model['name'] = self.car_model.keys()[0]
            depth, mask = self.render_car(
                    car_model['pose'], car_model['name'])
            # image = uts.drawboxes(self.image, [car_model['box']])
            self.image[:, :, 0] = mask * 255
            uts.plot_images({'image':self.image[:, :, ::-1],
                'depth': depth,
                'mask': mask})

            self.masks, self.depth, _, is_valid = \
                    eval_uts.merge_inst(
                        {'depth': depth, 'mask': mask}, inst_id + 1,
                        self.masks,
                        self.depth,
                        thresh=0)

    self.valid_id = np.unique(self.masks)
    self.valid_id = self.valid_id[self.valid_id > 0]

    label_c = uts.label2color(self.masks, self.data_params['color_map_list'],
            [255,255,255])
    alpha = 0.7
    image = np.uint8(alpha * self.image + (1-alpha) * label_c);
    uts.plot_images({'image': image,
                     'mask': self.masks,
                     'depth': self.depth})
    return True



if __name__ == '__main__':
    """ visualize a depth example and evaluate the depth estimation results
    """
    # vis_example(2, 1)
    # eval_results(data='kitti')
    # eval_results_xroad()
    # vis_xroad_depth()
    # vis_apollo_render_depth()
    # vis_depth(2, method=['psm'])
    vis_apollo_depth('180116_053947113_Camera_5')




