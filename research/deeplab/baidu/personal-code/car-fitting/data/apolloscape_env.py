import os
import car_models
import scipy.io as io
import numpy as np
import extern.render.render as render
import evaluation.eval_utils as eval_uts
from random import randint
import scipy.ndimage.morphology as mo
import utils.utils as uts
import utils.utils_3d as uts_3d
import utils.transforms as trs
import cv2

import data.data_setting as data_setting
import data.kitti as kitti
import data.apolloscape as apollo
import preprocessor as ps
from collections import OrderedDict
import logging
import pdb

DTYPE = np.float32
np.set_printoptions(threshold=np.nan)

data_libs = {}
data_libs['kitti'] = kitti
data_libs['apollo'] = apollo

class Env_s(object):
    def __init__(self, c, data_params, split='train'):
        self.data_params = data_params
        self.split = split
        self.image_list = [image_name.strip() for image_name in open(
             data_params[split + '_list'])]

        if 'pid' in dir(c) and 'nprocs' in dir(c):
            self.image_list = self.image_list[c.pid::c.nprocs]

        # random perturb later
        self.list_order = range(len(self.image_list))
        self.image_num = len(self.image_list)
        self.counter = 0
        self.inst_counter = 0
        self.inst_counter_g = 0
        self.timestep_limit = c.timestep_limit
        self.config = c

        self.floor_plane = 1.0
        self.base = 1.3 / 1.6
        self.action_dim = 6

        # balance for reward function between IOU and rel
        self.alpha = 0.3

        # threshold for occlusion, if larger than 3.5
        self.occ_thr = 2

        # image scale for smaller images
        self.scale = c.scale
        h, w = self.data_params['size']

        # for the renderer problem
        hs, ws = np.uint32(self._round_to_even(
                 np.float32([h * self.scale, w * self.scale])))
        self.image_size = [hs, ws]
        self.image_size_ori = [h, w]
        self.init_rot = np.array([0.0, np.pi/2.0, np.pi, 3 * np.pi/2.0])

        self.state_names = ['image', 'depth', 'mask', 'pose']
        self.action_names = ['del_pose']
        self.reward_names = ['reward']

        self.state_update_names = ['pose'] # change during the steps
        self.inspector_names = ['IOU', 'delta']
        self.state = OrderedDict({})

        self.with_disc = False
        if 'bins' in data_params:
            self.bins = data_params['bins']
            self.with_disc = True

        if self.config.is_crop:
            self.setting = data_setting.get_policy_data_setting(self)


    def get_pose_image(self):

        if self.counter == self.image_num:
            self.counter = 0
            self.inst_counter_g = 0
            self.list_order = np.random.permutation(self.image_num)

        idx = self.list_order[self.counter]
        image_name = self.image_list[idx]
        self.image_name = image_name[:-4]

        image_path = self.data_params['image_path']
        state = OrderedDict({})
        state['image'] = data_libs[self.d_name].crop_image_read(
                         image_path + self.image_name + '.jpg')
        state['mask'] = data_libs[self.d_name].crop_image_read(
                self.data_params['car_mask_path'] + self.image_name + '.png')

        # self.depth = data_libs[self.d_name].cnn_depth_read(
        #         self.data_params['car_depth_path'] + self.image_name + '.png')
        # self.depth = data_libs[self.d_name].crop_image_read(self.depth)

        # generate depth & mask from the given poses
        pose_file = self.data_params['car_pose_path'] + self.image_name + '.poses'
        self.car_poses = data_libs[self.d_name].read_carpose(
                pose_file, is_euler=False)
        # theshold for valid mask
        state['pose_map'] = uts.poses_to_map(self.car_poses, self.mask)

        return state



class Env(object):
    def __init__(self, c, data_params, split='train', dataname='apollo'):
        self.data_params = data_params
        self.d_name = dataname
        self.split = split
        self.image_list = [image_name.strip() for image_name in open(
             data_params[split + '_list'])]

        if 'pid' in dir(c) and 'nprocs' in dir(c):
            self.image_list = self.image_list[c.pid::c.nprocs]

        # random perturb later
        self.list_order = range(len(self.image_list))
        self.image_num = len(self.image_list)
        self.counter = 0
        self.inst_counter = 0
        self.inst_counter_g = 0
        self.timestep_limit = c.timestep_limit
        self.config = c

        self.floor_plane = 1.0
        self.base = 1.3 / 1.6
        self.action_dim = 6

        # balance for reward function between IOU and rel
        self.alpha = 0.3

        # threshold for occlusion, if larger than 3.5
        self.occ_thr = 2

        # image scale for smaller images
        self.scale = c.scale
        h, w = self.data_params['size']

        # for the renderer problem
        hs, ws = np.uint32(self._round_to_even(
             np.float32([h * self.scale, w * self.scale])))
        self.image_size = [hs, ws]
        self.image_size_ori = [h, w]
        self.init_rot = np.array([0.0, np.pi/2.0, np.pi, 3 * np.pi/2.0])

        self.car_model = OrderedDict({})
        for name, car_type in data_params['car_names']:
            car_file = '%s/%s/%s' % (data_params['car_model_path'], car_type, name)
            self.car_model[name] = data_libs[dataname].load_car(car_file)
            self.car_model[name]['car_type'] = car_type

        self.state_names = ['image', 'depth', 'mask', 'pose']
        self.action_names = ['del_pose']
        self.reward_names = ['reward']

        self.state_update_names = ['pose'] # change during the steps
        self.inspector_names = ['IOU', 'delta']
        self.state = OrderedDict({})

        self.with_disc = False
        if 'bins' in data_params:
            self.bins = data_params['bins']
            self.with_disc = True

        if self.config.is_crop:
            self.setting = data_setting.get_policy_data_setting(self)


    def _round_to_even(self, num):
        return np.ceil(num / 4.) * 4.


    def intr_after_crop(self, intrinsic, crop):
        intr = uts_3d.intrinsic_mat_to_vec(intrinsic)
        intr[2] = intr[2] - crop[1] * self.image_size[1]
        intr[3] = intr[3] - crop[0] * self.image_size[0]
        return intr


    def padding_image(self, image_in, crop, interpolation=cv2.INTER_NEAREST):
        image = image_in.copy()
        if np.ndim(image) == 2:
            image = image[:, :, None]
        # pdb.set_trace()
        dim = image.shape[2]
        image_pad = np.zeros(self.image_size + [dim], dtype=image_in.dtype)
        h, w = self.image_size
        crop_cur = np.uint32([crop[0] * h, crop[1] * w, crop[2] * h, crop[3] * w])
        image = cv2.resize(
                image, (crop_cur[3] - crop_cur[1], crop_cur[2] - crop_cur[0]),
                interpolation=interpolation)
        image = image[:, :, None] if np.ndim(image) == 2 else image
        image_pad[crop_cur[0]:crop_cur[2], crop_cur[1]:crop_cur[3], :] = image

        if np.ndim(image) == 3:
            image_pad = np.squeeze(image_pad)
        return image_pad


    def save_rect_image_set(self):
        camera_name = 'Camera_5'
        self.intrinsic = uts_3d.intrinsic_vec_to_mat(
              self.data_params[camera_name + '_intr'], self.image_size)
        rect_mat = np.eye(4)
        rect_mat[:3, :3] = self.data_params[camera_name + '_rot']

        # adjust intrinsic
        hs, ws = self.image_size
        for image_name in self.image_list:
            print 'processing %s' % image_name
            image_name = image_name[:-4]
            cnn_depth_file = self.data_params['depth_path_rect'] + image_name + '.png'
            if not os.path.exists(cnn_depth_file):
                print cnn_depth_file + ' not exist'
                continue
            image = cv2.imread(self.data_params['image_path'] + image_name + '.jpg')
            mask_gt_ori = cv2.imread(
                   self.data_params['car_inst_path'] + image_name + '.png',
                   cv2.IMREAD_UNCHANGED)

            cnn_depth = data_libs[self.d_name].cnn_depth_read(
                   self.data_params['depth_path_rect'] + image_name + '.png')
            cnn_depth = self.padding_image(cnn_depth, self.data_params['rect_crop'])
            # pose_file = self.data_params['car_pose_path'] + \
            #      image_name + '.poses'

            image_rect = data_libs[self.d_name].rect_image(image, params, camera_name)
            mask_gt = data_libs[self.d_name].rect_image(
                    mask_gt_ori, params, camera_name, cv2.INTER_NEAREST)

            # self.car_poses = data_libs[self.d_name].read_carpose(
            #        pose_file, is_euler=False)
            image_rect = cv2.resize(image_rect, (ws, hs))
            mask_gt = cv2.resize(mask_gt, (ws, hs), interpolation=cv2.INTER_NEAREST)
            # valid_id = np.unique(mask_gt)
            # valid_id = valid_id[valid_id > 0]

            # intr = uts_3d.intrinsic_mat_to_vec(self.intrinsic)
            # masks = uts.label2mask(mask_gt)
            # floor_mask = ps.get_floor_mask(cnn_depth, intr, floor_height=3.3,
            #         rescale=0.25)
            # uts.plot_images({'floor_mask': floor_mask})
            # masks = ps.denoise_mask(masks, cnn_depth, floor_mask)
            # mask_gt_new = uts.mask2label(masks, valid_id)

            # for car_model in self.car_poses:
            #     pose = np.matmul(rect_mat, car_model['pose'])
            #     depth, mask = self.render_car(pose, car_model['name'])
            #     image_rect[:, :, 0] = mask * 255
            #     uts.plot_images({'image':image_rect[:, :, ::-1],
            #         'image_ori': image[:, :, ::-1],
            #         'depth': depth,
            #         'mask': mask})
            image_crop = data_libs[self.d_name].crop_image(image_rect,
                    self.data_params['rect_crop'])
            mask_gt_crop = data_libs[self.d_name].crop_image(mask_gt,
                    self.data_params['rect_crop'])
            # intr = self.intr_after_crop(self.intrinsic, self.data_params['rect_crop'])

            # uts.plot_images({'image':image_crop[:, :, ::-1],
            #                  'image_rect': image_rect[:, :, ::-1],
            #                  # 'floor_mask': floor_mask,
            #                  'depth': cnn_depth,
            #                  'mask_gt': mask_gt,
            #                  'mask_gt_crop': mask_gt_crop}, layout=[2, 3])
            # print np.unique(mask_gt_crop)
            # for idx in valid_id:
            #     cur_mask = mask_gt_crop == idx
            #     image_crop[:, :, 0] = cur_mask * 255
            #     uts.plot_images({'mask': cur_mask,  'image': image_crop[:, :, ::-1]})

            cv2.imwrite(self.data_params['image_path_rect'] + image_name + '.jpg',
                    image_crop)
            cv2.imwrite(self.data_params['car_inst_path_rect'] + image_name + '.png',
                    np.uint16(mask_gt_crop))


    def get_image(self):
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
        # mask_path = self.data_params['car_inst_path']
        # depth_path = self.data_params['depth_path_v2']
        # depth_file = depth_path + self.image_name + '.png'
        # mask_file = mask_path + self.image_name + '.png'
        # if os.path.exists(depth_file):
        #     self.depth = data_libs[self.d_name].depth_read(depth_file)
        #     self.masks = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        #     gen_depth = False

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
            self.depth = 10000. * np.ones(self.image_size)
            self.masks = np.zeros(self.depth.shape)

            for inst_id in range(len(self.car_poses)):
                car_model = self.car_poses[inst_id]
                # if not (car_model['name'] in self.car_model):
                #     car_model['name'] = self.car_model.keys()[0]
                depth, mask = self.render_car(
                        car_model['pose'], car_model['name'])

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

        # mask_file = self.data_params['car_inst_path'] + self.image_name + '.png'
        # cv2.imwrite(mask_file, self.total_mask.astype(np.uint8))
        # self.total_depth[self.total_depth == 10000] = 0.0
        # save_depth = np.uint16(self.total_depth * 100.0)
        # depth_file = self.data_params['depth_path_v2'] + self.image_name + '.png'
        # cv2.imwrite(depth_file, save_depth.astype(np.uint16))
        # logging.info('save %s depth %s' % (mask_file, depth_file))
        # saved_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        # saved_depth = np.float32(save_depth) / 100.0
        # saved_mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)

        # label_c = uts.label2color(self.masks, self.data_params['color_map_list'],
        #         [255,255,255])
        # alpha = 0.7
        # image = np.uint8(alpha * self.image + (1-alpha) * label_c);
        # uts.plot_images({'image': image,
        #                  'mask': self.masks,
        #                  'depth': self.depth})
        return True


    def get_croped_size(self, sz, crop):
        crop_new = crop.copy()
        h, w = sz
        crop_new[[0, 2]] *= h
        crop_new[[1, 3]] *= w
        crop_new = np.uint32(crop)
        return crop_new[2] - crop_new[0], crop_new[3] - crop_new[1]


        if self.config.is_crop:
            crop = uts.get_mask_bounding_box(self.state['mask'], context=0.5)
            self.state['crop'] = crop
            for i, name in enumerate(['render_depth', 'image', 'depth', 'mask']):
                if name in self.state:
                    temp = uts.crop_image(self.state[name], crop)
                    inter = self.setting[name]['interpolation']
                    self.state[name], pad = uts.resize_and_pad(
                                np.float32(temp),
                                tuple([self.config.height, self.config.width]),
                                interpolation=inter,
                                get_pad_sz=True)
                    self.state['pad'] = pad

            # print action, self.state['pose'], self.state['crop']
            # uts.plot_images({'image': np.uint8(self.state['image']),
            #                  'mask': self.state['mask'],
            #                  'depth': self.state['depth'],
            #                  'render_depth': self.state['render_depth']})

        return self.state, action


    def get_image_rect(self):
        # sample an image with stereo rectification (so that we can use stereo depth \
        # for learning)
        if self.counter == self.image_num:
            self.counter = 0
            self.inst_counter_g = 0
            self.list_order = np.random.permutation(self.image_num)

        idx = self.list_order[self.counter]
        image_name = self.image_list[idx]
        self.image_name = image_name[:-4]

        image_path = self.data_params['image_path_rect']
        self.image = cv2.imread(image_path \
                   + self.image_name + '.jpg', cv2.IMREAD_UNCHANGED)

        for name in self.data_params['intrinsic'].keys():
            if name in image_name:
                self.cam_name = name
                break

        self.intrinsic = uts_3d.intrinsic_vec_to_mat(
                self.data_params[self.cam_name + '_intr'], self.image_size)
        self.focal_len = self.intrinsic[0, 0]

        mask_path = self.data_params['car_inst_path_rect']
        depth_path = self.data_params['depth_path_rect']

        depth_file = depth_path + self.image_name + '.png'
        mask_file = mask_path + self.image_name + '.png'
        if not os.path.exists(depth_file):
            return False

        self.depth = data_libs[self.d_name].cnn_depth_read(depth_file)
        self.masks = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        self.image = self.padding_image(self.image, self.data_params['rect_crop'],
                cv2.INTER_CUBIC)
        self.depth = self.padding_image(self.depth, self.data_params['rect_crop'])
        self.masks = np.uint16(
                self.padding_image(self.masks, self.data_params['rect_crop']))

        # sample an instance
        self.inst_counter = 0

        # generate depth & mask from the given poses
        pose_file = self.data_params['car_pose_path'] + self.image_name + '.poses'
        if not os.path.exists(pose_file):
            return False

        self.car_poses = data_libs[self.d_name].read_carpose(
                pose_file, is_euler=False)
        self.valid_id = np.unique(self.masks)

        # theshold for valid mask
        valid_mask = [(i > 0 and np.sum(self.masks == i) > 10) for i in self.valid_id]
        self.valid_id = self.valid_id[valid_mask]

        return True


    def sample_state_rect(self):
        if self.counter == 0:
            while not self.get_image_rect():
                self.counter += 1
                if self.counter == self.image_num:
                    raise ValueError('All images has no mask')
            self.counter += 1

        if self.inst_counter == len(self.valid_id):
            while not self.get_image_rect():
                self.counter += 1
            self.counter += 1

        # change to network input
        car_inst_id = self.valid_id[self.inst_counter]
        self.state['image_name'] = self.image_name
        self.state['inst_id'] = car_inst_id
        self.state['image'] = self.image.copy()[:, :, ::-1]
        self.state['depth'] = self.depth.copy()
        self.state['mask'] = self.masks == car_inst_id
        self.state['intrinsic'] = self.intrinsic

        car_model = self.car_poses[car_inst_id - 1]
        pose_gt = np.matmul(self.data_params[self.cam_name + '_ext'],
                car_model['pose'])

        # convert to euler angle ( rescale to -pi to pi)
        rot = uts_3d.rotation_matrix_to_euler_angles(pose_gt[:3, :3], check=False)
        # rot[2] += np.pi
        trans = pose_gt[:3, 3].flatten()
        pose_gt = np.hstack([rot, trans])
        self._centroid_pose = self.init_pose()

        # car_name = self.car_model.keys()[0]
        # depth, mask = self.render_car(self._centroid_pose[0], car_name,
        #         is_rect=True)
        # self.state['render_depth'] = depth
        self.state['pose'] = DTYPE(self._centroid_pose.copy())

        self.inst_counter += 1
        self.inst_counter_g += 1
        action = {'del_pose': pose_gt[None, :]}

        if self.with_disc:
            action['disc_pose'] = uts.find_bin_idx(pose_gt - self.state['pose'][0], \
                    self.bins, is_equal=True)[None, :]

        if self.with_points:
            self.state['point'] = uts_3d.depth2points()

        if self.config.is_crop:
            crop = uts.get_mask_bounding_box(self.state['mask'], context=0.5)
            self.state['crop'] = crop
            for i, name in enumerate(['render_depth', 'image', 'depth', 'mask']):
                if name in self.state:
                    temp = uts.crop_image(self.state[name], crop)
                    inter = self.setting[name]['interpolation']
                    self.state[name], pad = uts.resize_and_pad(
                                np.float32(temp),
                                tuple([self.config.height, self.config.width]),
                                interpolation=inter,
                                get_pad_sz=True)
                    self.state['pad'] = pad

            # print action, self.state['pose'], self.state['crop']
            # uts.plot_images({'image': np.uint8(self.state['image']),
            #                  'mask': self.state['mask'],
            #                  'depth': self.state['depth'],
            #                  'render_depth': self.state['render_depth']})

        return self.state, action



    def render_car(self, pose, car_name, is_rect=False):
        car = self.car_model[car_name]
        if not ('scale' in car):
            car['scale'] = np.ones((3, ))

        vert = uts_3d.project(pose, car['scale'], car['vertices'])
        intrinsic = np.float64(uts_3d.intrinsic_mat_to_vec(self.intrinsic))
        depth, mask = render.renderMesh_np(
                np.float64(vert),
                np.float64(car['faces']),
                intrinsic, self.image_size[0], self.image_size[1])
        mask = mask[::-1, :]
        depth = depth[::-1, :]

        return depth, mask


    def sample_state(self):
        if self.counter == 0:
            while not self.get_image():
                self.counter += 1
                if self.counter == self.image_num:
                    raise ValueError('All images has no mask')
            self.counter += 1

        if self.inst_counter == len(self.valid_id):
            while not self.get_image():
                self.counter += 1
            self.counter += 1

        # change to network input
        car_inst_id = self.valid_id[self.inst_counter]
        self.state['image_name'] = self.image_name
        self.state['inst_id'] = car_inst_id
        self.state['image'] = self.image.copy()[:, :, ::-1]
        self.state['depth'] = self.depth.copy()
        self.state['mask'] = self.masks == car_inst_id

        car_model = self.car_poses[car_inst_id - 1]
        self._centroid_pose = self.init_pose()

        car_name = self.car_model.keys()[0]
        depth, mask = self.render_car(self._centroid_pose[0], car_name)
        self.state['render_depth'] = depth
        self.state['pose'] = DTYPE(self._centroid_pose.copy())

        self.inst_counter += 1
        self.inst_counter_g += 1
        action = {'del_pose': car_model['pose'][None, :]}

        # pdb.set_trace()
        # print action, self.state['pose']
        # uts.plot_images({'image': self.state['image'],
        #                  'mask': self.state['mask'],
        #                  'depth': self.state['depth'],
        #                  'render_depth': self.state['render_depth']})
        if self.config.is_crop:
            crop = uts.get_mask_bounding_box(self.state['mask'], context=0.5)
            self.state['crop'] = crop
            for name in ['render_depth', 'image', 'depth', 'mask']:
                temp = uts.crop_image(self.state[name], crop)
                inter = self.setting[name]['interpolation']
                self.state[name] = uts.resize_and_pad(
                            np.float32(temp),
                            tuple([self.config.height, self.config.width]),
                            interpolation=inter)

            uts.plot_images({'image': np.uint8(self.state['image']),
                             'mask': self.state['mask'],
                             'depth': self.state['depth'],
                             'render_depth': self.state['render_depth']})

        return self.state, action


    def get_mcmc_image(self):
        # sample an image
        if self.counter == self.image_num:
            self.counter = 0
            self.inst_counter_g = 0
            self.list_order = np.random.permutation(self.image_num)

        idx = self.list_order[self.counter]
        image_name = self.image_list[idx]
        self.image_name = image_name

        disp = 'psm'
        res_path = self.data_params['data_path'] + \
                'displets_gt_seg/%s/' % disp
        cnn_disp_path = self.data_params['data_path'] + \
                '/dispmaps/%s/disp_0/' % disp
        disp_mat_file = res_path + image_name + '_car.mat'

        if not os.path.exists(disp_mat_file):
            return False

        displets = io.loadmat(disp_mat_file)
        inst_num = np.max(displets['mask'])

        if inst_num == 0:
            return False

        if self.scale != 1.0:
            displets['mask'] = cv2.resize(displets['mask'],
                    tuple(self.image_size[::-1]),
                    interpolation=cv2.INTER_NEAREST)

        self.masks = uts.one_hot(displets['mask'], inst_num + 1)[:, :, 1:]
        self.masks = np.transpose(self.masks, (2, 0, 1))
        self.poses = np.zeros((inst_num, 6))
        self.poses[:, [1, 3, 4, 5]] = displets['poses']
        self.poses[:, 4] = -1 * (self.poses[:, 4] + 1.6)

        self.image = cv2.imread(self.data_params['image_path'] \
                          + image_name + '.png', cv2.IMREAD_UNCHANGED)
        self.depth = kitti.depth_read(cnn_disp_path + image_name + '.png')

        intr_file = self.data_params['calib_path'] + \
                         image_name[:-3] + '.txt'
        self.intrinsic = kitti.load_intrinsic(intr_file, mat=True)
        self.focal_len = self.intrinsic[0, 0]

        # sample an instance
        self.inst_counter = 0
        if self.scale != 1.0:
            hs, ws = self.image_size
            h, w = self.image_size_ori
            self.image = cv2.resize(self.image, (ws, hs))
            self.depth = cv2.resize(self.depth, (ws, hs),
                    interpolation=cv2.INTER_NEAREST)
            self.intrinsic[:2, :] *= (np.float32(hs)/np.float32(h))

        return True


    def valid_pose(self, trans, all_trans, dis=4.):
        """ all_trans, list of 1 x 3 points
            trans, dim 3 point
        """
        # if all_trans.shape[0] > 8:
        #     pdb.set_trace()
        min_dis = np.min(np.sum(np.abs(all_trans - trans), axis=1))
        return min_dis > dis


    def simulate_image(self):
        sim_num = np.random.randint(5, 20)
        cur_counter = self.counter
        while not self.get_mcmc_image():
            self.counter += 1
            if self.counter % self.image_num == cur_counter:
                raise ValueError('All images has no mask')
        self.counter += 1

        self.total_depth = 10000. * np.ones(self.image_size)
        self.total_mask = np.zeros(self.image_size)
        self.poses = []
        self.mask_sizes = []

        intrinsic = np.float64(
                uts_3d.intrinsic_mat_to_vec(self.intrinsic))
        self.focal_len = self.intrinsic[0, 0]

        inst_id = 1
        step = 0
        while inst_id <= sim_num and step < 100:
            cur_pose = np.zeros((1, 6))
            pix = [np.random.randint(self.image_size[0] / 2,
                                     self.image_size[0]),
                   np.random.randint(0, self.image_size[1])]

            # sample a car id
            cur_pose[:, 3:] = self.pix2point(pix)
            if inst_id > 1:
                poses = np.vstack(self.poses)[:, 3:]
                if not self.valid_pose(cur_pose[:, 3:], poses, dis=5.):
                    step += 1
                    continue

            car_id = np.random.randint(0, len(self.car_model.keys()))
            car_name, car = self.car_model.items()[car_id]
            idx = np.random.randint(4)
            cur_pose[:, 1] = self.init_rot[idx]

            vert = uts_3d.project(
                    cur_pose[0], car['scale'], car['vertices'])
            depth, mask = render.renderMesh_np(np.float64(vert),
                    np.float64(car['faces']),
                    intrinsic, self.image_size[0], self.image_size[1])

            # logging.info('%s' % cur_pose)
            # uts.plot_images({'mask': mask,
            #                  'depth': depth})
            self.total_mask, self.total_depth, _, is_valid = \
                    eval_uts.merge_inst(
                    {'depth': depth, 'mask': mask}, inst_id,
                    self.total_mask,
                    self.total_depth,
                    thresh=0.3) # at lease see 30% of the render mask

            # uts.plot_images({'depth': self.total_depth,
            #     'mask': self.total_mask})
            if is_valid:
                self.poses.append(cur_pose)
                self.mask_sizes.append(np.sum(mask))
                step += 1
                inst_id += 1
            else:
                step += 1
                continue

        self.inst_id = inst_id - 1
        bkg_mask = self.total_depth == 10000.
        # self.total_depth[bkg_mask] = self.depth[bkg_mask]
        self.total_depth[bkg_mask] = 300.

        # uts.plot_images({'mask': self.total_mask,
        #                  'depth': self.total_depth})


    def simulate_state_from_image(self, is_rand=False):

        if self.inst_counter == 0:
            self.simulate_image()

        self.state['image'] = np.zeros(self.image_size + [3])
        is_good = False
        while not is_good:
            self.state['mask'] = self.total_mask == (self.inst_counter + 1)
            portion = np.sum(self.state['mask']) / \
                    self.mask_sizes[self.inst_counter]
            if portion > 0.15:
                is_good = True
            else:
                self.inst_counter += 1

        pose_gt = self.poses[self.inst_counter]

        mask = self.state['mask']
        # self.state['depth'] = -1 * np.ones(self.image_size)
        # self.state['depth'][mask] = self.total_depth[mask]
        self.state['depth'] = self.total_depth

        self.state['init_pose'] = self.init_pose(is_rand=is_rand)
        self.state['pose'] = self.state['init_pose']

        depth, mask = self.render_car(self.state['init_pose'][0], 0)
        self.state['render_depth'] = depth
        self.inst_counter += 1
        if self.inst_counter == self.inst_id:
            self.inst_counter = 0

        # logging.info(pose_gt)
        # logging.info(self.state['init_pose'])
        # uts.plot_images({'mask': self.state['mask'],
        #          'depth': self.state['depth'],
        #          'render_depth': self.state['render_depth']})
        # del_pose = pose_gt - self.state['pose']
        del_pose = pose_gt
        return self.state, {'del_pose': del_pose}


    def simulate_state(self):
        """ For each image we are simuating 10 masks
        """
        if self.counter == 0:
            while not self.get_mcmc_image():
                self.counter += 1
                if self.counter == self.image_num:
                    raise ValueError('All images has no mask')
            self.counter += 1

        if self.inst_counter == len(self.masks):
            while not self.get_mcmc_image():
                self.counter += 1
            self.counter += 1

        # sample pose
        self.state['image'] = np.zeros(self.image.shape)
        pix = [np.random.randint(self.image_size[0] / 2,
                                 self.image_size[0]),
               np.random.randint(0, self.image_size[1])]

        self.state['mask'] = np.zeros(self.image_size)
        t = max(0, pix[0] - 10)
        b = min(self.image_size[0], pix[0] + 10)
        l = max(0, pix[1] - 10)
        r = min(self.image_size[1], pix[1] + 10)
        self.state['mask'][t:b, l:r] = 1

        # sample a car id
        self.state['depth'] = self.depth
        cur_pose = self.init_pose(is_rand=True)
        cur_pose[:, 1] = self.init_rot[np.random.randint(4)]
        # cur_pose = np.zeros((1, 6))
        # idx = np.random.randint(4)
        # cur_pose[:, 1] = self.init_rot[idx]
        # cur_pose[:, [3, 4, 5]] = np.array([0.0, -1.6, 10.0])

        # observations
        # car_id = np.random.randint(0, len(self.car_model.keys()))
        car_id = 0
        depth, mask = self.render_car(cur_pose[0], car_id)
        self.state['depth'] = depth
        self.state['mask'] = mask

        # initial guess
        self.state['pose'] = self.init_pose(is_rand=False)
        # print cur_pose
        depth, mask = self.render_car(self.state['pose'][0], 1)
        self.state['render_depth'] = depth

        # gt pose
        # action = {'del_pose': cur_pose - self.state['pose']}
        action = {'del_pose': cur_pose}

        # print action, cur_pose, self.state['pose']
        # uts.plot_images({'mask': self.state['mask'],
        #                  'depth': self.state['depth'],
        #                  'render_depth': self.state['render_depth']})

        return self.state, action


    def sample_mcmc_state(self):
        """ sample from fitted results using mcmc
        """
        if self.counter == 0:
            while not self.get_mcmc_image():
                self.counter += 1
                if self.counter == self.image_num:
                    raise ValueError('All images has no mask')
            self.counter += 1

        if self.inst_counter == len(self.masks):
            while not self.get_mcmc_image():
                self.counter += 1
            self.counter += 1

        if self.inst_counter == 0:
            # change to network input
            self.state['image'] = self.image.copy()

        self.state['mask'] = self.masks[self.inst_counter].copy() > 0
        # mask = self.state['mask']
        # self.state['depth'] = -1 * np.ones(self.image_size)
        # self.state['depth'][mask] = self.depth[mask]
        self.state['depth'] = self.depth

        # direct regression without consider
        pose = self.poses[self.inst_counter].copy()[None, :]
        # self.state['pose'] = np.zeros((1, self.action_dim))
        # action = {'del_pose': pose}

        self._centroid_pose = self.init_pose()
        self.state['pose'] = DTYPE(self._centroid_pose.copy())
        depth, mask = self.render_car(self.state['pose'][0], 0)
        self.state['render_depth'] = depth

        # action = {'del_pose': pose - self.state['pose']}
        action = {'del_pose': pose}
        # logging.info('%s' % mcmc_pose)
        # uts.plot_images({'mask': self.state['mask'],
        #                  'depth': self.state['depth']})

        self.inst_counter += 1
        self.inst_counter_g += 1

        return self.state, action


    def pix2point(self, pix, depth=None):
        if depth is None:
            z = self.depth[pix[0], pix[1]]
        else:
            z = depth[pix[0], pix[1]]

        x = np.float32([pix[1], pix[0], 1.0]) * z
        x = np.dot(np.linalg.inv(self.intrinsic), x)
        assert np.linalg.norm(x) != 0
        v = x / np.linalg.norm(x)
        x = x + v
        # x[1] = -1.0 * (x[1] + self.floor_plane)
        return x


    def init_pose(self, is_rand=False, state=None):
        """ calculate pose in original image
        """

        has_state = not (state is None)
        def eros_mask(mask):
            area = np.float32(np.sum(mask))
            ero_iter = np.uint8(max(area / (4 * np.sqrt(area)), 1))
            mask = mo.binary_erosion(mask, iterations=ero_iter)
            return mask

        if not has_state:
            mask = self.masks == self.state['inst_id']
            # mask_sm = eros_mask(mask)
            mask_sm = np.logical_and(mask, self.depth> 0)
            mask_sm = mask_sm if np.sum(mask_sm) > 0 else \
                        np.logical_and(mask, self.depth> 0)
            depth = self.depth
            intr = self.intrinsic
        else:
            mask = state['mask']
            mask_sm = np.logical_and(mask, state['depth']> 0)
            depth = state['depth']
            intr = state['intrinsic']

        pose = np.zeros((1, 6), dtype=np.float32)
        if np.sum(mask_sm) == 0:
            return pose

        # sample translation
        # change to median depth value for init
        pixs = np.where(mask_sm)
        pixs = np.vstack(pixs)

        if not is_rand:
            depths= depth[pixs[0, :], pixs[1, :]]
            median_idx = np.argsort(depths)[len(depths) // 2]
            pix = pixs[:, median_idx]
        else:
            idx = randint(0, pixs.shape[1] - 1)
            pix = pixs[:, idx]

        if has_state:
            z = depth[pix[0], pix[1]]
            pix = (pix - state['pad'][:2]) / state['pad'][-1] + \
                   state['crop'][:2]
            x = np.float32([pix[1], pix[0], 1.0]) * z
            x = np.dot(np.linalg.inv(intr), x)
            assert np.linalg.norm(x) != 0
            v = x / np.linalg.norm(x)
            x = x + v
            pose[:, 3:] = x
        else:
            pose[:, 3:] = self.pix2point(pix)

        pose[:, 2] = -1 * np.pi
        # sample rotation from one of the 4th init
        if is_rand:
            idx = np.random.randint(4)
            pose[:, 1] = self.init_rot[idx]

        return pose


    def step(self, act, cur_state=None, get_res=False, is_plot=False):
        """ reward_name: what kind of reward to use
                  'mcmc': the reward function from CVPR 2015
        """

        # metric to check whether the fitting is good
        reward = OrderedDict({})
        inspector = OrderedDict({})
        reward_name = self.config.reward_name

        reward['reward'] = np.zeros((1, len(self.car_model)))
        IOUs, deltas = [np.zeros(len(self.car_model)) for i in range(2)]
        done = False

        if cur_state is None:
            next_pose = DTYPE(self.state['pose'] + act['del_pose'])
            cur_mask = DTYPE(self.state['mask'])
            cur_depth = DTYPE(self.state['depth'])
        else:
            next_pose = DTYPE(cur_state['pose'] + act['del_pose'])
            cur_mask = cur_state['mask']
            cur_depth = cur_state['depth']
            if 'intrinsic' in cur_state.keys():
                self.intrinsic = cur_state['intrinsic']
                self.focal_len = self.intrinsic[0, 0]

        cur_depth[cur_depth <= 0.] = -1.0
        # logging.info('%s %s' % ('act', act['del_pose']))
        # logging.info('%s %s' % ('pose', self.state['pose']))

        # we don't consider rotation up and side, depth must > 0.1
        next_pose[:, 5] = max(0.5, next_pose[:, 5])
        # render has 0 at left bottom
        # next_pose[:, 4] = -1.0 * next_pose[:, 4]

        max_reward = -1 * np.inf
        res = {}

        for i, (car_name, car) in enumerate(self.car_model.items()):

            depth, mask = self.render_car(next_pose[0], car_name)

            # state could be ok
            if self.config.is_crop:
                cur_crop = self.state['crop'] if cur_state is None \
                        else cur_state['crop']
                res = {'mask': mask, 'depth': depth}
                for name in res.keys():
                    temp = uts.crop_image(res[name], cur_crop)
                    inter = self.setting[name]['interpolation']
                    res[name] = uts.resize_and_pad(
                                np.float32(temp),
                                tuple([self.config.height, self.config.width]),
                                interpolation=inter)
                mask, depth = res['mask'], res['depth']

            if is_plot and i == 0:
                logging.info('%s' % next_pose)
                uts.plot_images({'mask': mask,
                                 'depth': depth,
                                 'mask_in': cur_mask,
                                 'depth_in': cur_depth})
                # raw_input("Press the <ENTER> key to continue...")

            if reward_name == 'mine':
                IOU, delta, reward['reward'][0, i] = \
                        eval_uts.compute_reward(cur_mask, cur_depth,
                                            mask, depth, self.occ_thr)
                IOUs[i] = IOU
                deltas[i] = delta

                # logging.info('%s, %s' %(IOU, delta))
                if IOU <= 1e-9:
                    if np.sum(mask) > 0:
                        center1 = uts.get_mask_center(cur_mask) \
                                / self.image_size
                        center2 = uts.get_mask_center(mask) \
                                / self.image_size
                        reward['reward'][0, i] += np.exp(
                            -30 * np.sum(np.square(center1 - center2)))

            elif reward_name == 'mcmc':
                cur_disp = uts_3d.depth2disp(cur_depth, self.focal_len, self.base)
                disp = uts_3d.depth2disp(depth, self.focal_len, self.base)
                IOU, delta, reward['reward'][0, i] = eval_uts.mcmc_reward(
                        cur_mask, cur_disp, mask, disp)
                IOUs[i] = IOU
                deltas[i] = delta


            elif reward_name == 'iou':
                cur_disp = uts_3d.depth2disp(cur_depth, self.focal_len, self.base)
                disp = uts_3d.depth2disp(depth, self.focal_len, self.base)
                IOU, delta, reward['reward'][0, i] = eval_uts.mcmc_reward(
                                            cur_mask, cur_disp, mask, disp)
                reward['reward'][0, i] = IOU
                IOUs[i] = IOU
                deltas[i] = delta

            else:
                raise ValueError('no given reward %s' % reward_name)

            if get_res:
                if max_reward < reward['reward'][0, i]:
                    max_reward = reward['reward'][0, i]
                    res.update({'mask': mask, 'depth': depth})

            if IOU > 0.9 and delta > 0.9:
                # IOU, delta, reward['reward'][0, i] = eval_uts.mcmc_reward(
                #   cur_mask, cur_disp, mask, disp)
                # uts.plot_images({'mask': mask,
                #                  'depth': depth,
                #                  'mask_in': cur_mask,
                #                  'depth_in': cur_depth})
                done = True

        # logging.info('%s %s' % ('reward', reward['reward']))
        reward['reward'] = DTYPE(reward['reward'])
        inspector['IOU'] = IOUs
        inspector['delta'] = deltas

        if get_res:
            return reward, done, {'pose': next_pose}, inspector, res
        else:
            return reward, done, {'pose': next_pose}, inspector


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
        depth_path = self.data_params['depth_path_v2']
        depth_file = depth_path + self.image_name + '.png'
        mask_path = self.data_params['car_inst_path']
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


    def set_stat_plot(self, set_names=['train', 'minival']):
        depth_all = []
        for data_name in set_names:
            depth_cur = []
            image_list = [image_name.strip()[:-4] for image_name in open(
                self.data_params[data_name + '_list'])]

            for i, image_name in enumerate(image_list):
                if i % 100 == 0:
                    print "%d / %d" % (i, len(image_list))
                mask_path = self.data_params['car_inst_path_rect']
                depth_path = self.data_params['depth_path_rect']
                depth_file = depth_path + image_name + '.png'
                mask_file = mask_path + image_name + '.png'
                depth = data_libs[self.d_name].depth_read(depth_file)
                masks = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
                valid_id = np.unique(masks)
                valid_id = valid_id[valid_id > 0]
                for idx in valid_id:
                    cur_mask = masks == idx
                    pixs = np.where(cur_mask)
                    pixs = np.vstack(pixs)

                    depths= depth[pixs[0, :], pixs[1, :]]
                    median_depth = np.sort(depths)[len(depths) // 2]
                    depth_cur.append(median_depth)
            depth_all.append(np.array(depth_cur))

        uts.plot_histogram(depth_all, labels=set_names)

    def resave_images(self):
        for i, image_name in enumerate(self.image_list):
            if i % 10 == 1:
                print i, '/', len(self.image_list)
            image_file = '%s/%s' % (self.data_params['image_path'], image_name)
            image = cv2.imread(image_file)
            assert image.shape[0] > 0
            out_image_file = '%s/%s' % (self.data_params['save_image_path'], image_name)
            cv2.imwrite(out_image_file, image)

    def random_sample(self, car_poses):
        car_poses_sim = [i for i in car_poses]
        pose_num = len(car_poses_sim)
        index = np.random.randint(pose_num, size=pose_num).tolist()
        car_poses_sim = car_poses_sim + [car_poses_sim[idx] for idx in index]

        for car_pose in car_poses_sim:
            # simulate score
            car_pose['score'] = float(np.random.rand(1)[0] / 2 + 0.4)

            # simulate noise
            trans_i = car_pose['pose'][3:]
            rot_i = car_pose['pose'][:3]
            trans_i, rot_p = uts_3d.random_perturb(trans_i, np.pi * np.ones(3) / 2,
                3.0, np.pi / 8.)
            rot_i[1] += (rot_p[0] - np.pi/2)
            car_pose['pose'] = np.concatenate([rot_i, trans_i])

        return car_poses_sim


    def simulate_noisy_pose(self):
        import json
        for i, image_name in enumerate(self.image_list):
            if i % 10 == 1:
                print i, '/', len(self.image_list)

            pose_file = self.data_params['car_pose_path'] + \
                    image_name[:-4] + '.poses'
            car_poses = data_libs[self.d_name].read_carpose(
                    pose_file, is_euler=True)
            car_poses = self.random_sample(car_poses)
            for car_pose in car_poses:
                if not isinstance(car_pose['pose'], list):
                    car_pose['pose'] = car_pose['pose'].tolist()

            save_path = self.data_params['car_pose_path_sim']
            pose_file_out = save_path + image_name[:-4] + '.json'
            with open(pose_file_out, 'w') as f:
                json.dump(car_poses, f, sort_keys=True, indent=4,
                    ensure_ascii=False)


    def check_resave_new_label(self):
        import json
        car_params = data_libs[self.d_name].set_params_car(self.split)
        image_list = [image_name[:-4] for image_name in \
                        sorted(os.listdir(car_params['car_pose_path']))]

        image_num = len(image_list)
        for i, image_name in enumerate(image_list):
            save_path = car_params['car_pose_path_json']
            pose_file_out = save_path + image_name + '.json'
            with open(pose_file_out) as f:
                car_poses_json = json.load(f)
            car_poses = np.loadtxt('%s/%s.txt' % (car_params['car_pose_path'],
                image_name), ndmin=2)

            if len(car_poses) - len(car_poses_json) >= 1:
                temp1 = [car_pose[1:].tolist() for car_pose in car_poses]
                temp2 = [car_pose['pose'] for car_pose in car_poses_json]
                temp3 = [car_pose for car_pose in temp1 if car_pose not in temp2]
                print temp3
            else:
                continue

            logging.info('%d / %d, %s' % (i, image_num, image_name))
            for name in self.data_params['intrinsic'].keys():
                if name in image_name:
                    cam_name = name
                    break

            self.intrinsic = uts_3d.intrinsic_vec_to_mat(
                    self.data_params['intrinsic'][cam_name], self.image_size)
            self.focal_len = self.intrinsic[0, 0]

            image = cv2.imread('%s/%s.jpg' % (car_params['image_path'], image_name),
                cv2.IMREAD_UNCHANGED)
            hs, ws = self.image_size
            image = cv2.resize(image, (ws, hs))

            car_poses = np.loadtxt('%s/%s.txt' % (car_params['car_pose_path'],
                image_name), ndmin=2)
            total_depth = 10000. * np.ones(self.image_size)
            total_mask = np.zeros(self.image_size)

            vis_rate = np.zeros(len(car_poses))
            for i, car_pose in enumerate(car_poses):
                car_name = car_models.car_id2name[int(car_pose[0])].name
                pose_new = car_pose[1:]
                # if pose_new[-1] < 1.5:
                #     continue

#                 if image_name == '180118_071316772_Camera_5' and i == 2:
#                     continue

                # print pose_new
                depth, mask = self.render_car(pose_new, car_name)
                total_mask, total_depth, _, is_valid = \
                         eval_uts.merge_inst(
                            {'depth': depth, 'mask': mask}, i + 1,
                            total_mask,
                            total_depth,
                            thresh=0)
                vis_rate[i] = np.float32(np.sum(mask))
                # uts.plot_images({'image': image, 'mask': mask, 'depth': depth,
                #     'total_mask': total_mask})

            uts.plot_images({'image': image, \
                    'total_mask': total_mask, 'depth': total_depth})

            # car_pose_json = []
            # for i, car_pose in enumerate(car_poses):
            #     car_pose_dict = {}
            #     cur_area = np.sum(total_mask == (i + 1))
            #     area = np.round(
            #         np.float32(cur_area) / (self.scale ** 2))
            #     vis_rate[i] = cur_area / (vis_rate[i] + np.spacing(1))

            #     depth = car_pose[-1]
            #     if area == 0:
            #         print i, 'rendered mask has 0 area'
            #         continue

            #     car_pose_dict['car_id'] = int(car_pose[0])
            #     car_pose_dict['pose'] = car_pose[1:].tolist()
            #     car_pose_dict['area'] = int(area)
            #     car_pose_dict['visible_rate'] = float(vis_rate[i])
            #     car_pose_json.append(car_pose_dict)

            # with open(pose_file_out, 'w') as f:
            #     json.dump(car_pose_json, f, sort_keys=True, indent=4,
            #         ensure_ascii=False)



    def resave_new_label(self, image_list):
        import json
        car_params = data_libs[self.d_name].set_params_car(self.split)

        # test_file_name = '171206_034630104_Camera_5'
        image_num = len(image_list)

        for i, image_name in enumerate(image_list):
            save_path = car_params['car_pose_path_json']
            pose_file_out = save_path + image_name + '.json'
            if os.path.exists(pose_file_out) and 'Camera_5' in image_name:
                continue

            logging.info('%d/%d, %s' % (i, image_num, image_name))
            cam_name = 'Camera_5'
            self.intrinsic = uts_3d.intrinsic_vec_to_mat(
                    self.data_params['intrinsic'][cam_name],
                    self.image_size)
            self.focal_len = self.intrinsic[0, 0]

            image = cv2.imread('%s/%s.jpg' % (car_params['image_path'], image_name),
                cv2.IMREAD_UNCHANGED)
            hs, ws = self.image_size
            image = cv2.resize(image, (ws, hs))

            if not os.path.exists('%s/%s.txt' % (car_params['car_pose_path'],
                image_name)):
                continue

            car_poses = np.loadtxt('%s/%s.txt' % (car_params['car_pose_path'],
                image_name), ndmin=2)

            total_depth = 10000. * np.ones(self.image_size)
            total_mask = np.zeros(self.image_size)

            vis_rate = np.zeros(len(car_poses))
            for i, car_pose in enumerate(car_poses):
                car_name = car_models.car_id2name[int(car_pose[0])].name
                if car_pose[-1] < 1.0:
                    continue
                    # angle = car_pose[1:4].copy()
                    # val = -1 * np.sign(angle) * np.pi
                    # angle[0] = -1 * (np.pi + angle[0])
                    # angle[1] = -1 * (np.pi + angle[1])
                    # angle[[2]] += np.pi
                    # angle[[1]] *= -1
                    # rot_mat = uts_3d.euler_angles_to_rotation_matrix(angle)
                    # rot_mat = np.linalg.inv(rot_mat)
                    # angle = uts_3d.rotation_matrix_to_euler_angles(rot_mat)
                    # trans = -1 * car_pose[4:]
                    # pose_new = np.hstack([angle, trans])
                else:
                    pose_new = car_pose[1:]

                # pose_new = car_pose[1:]
                # angle = car_pose[1:4].copy()
                # angle[-1] += np.pi
                # rot_mat = uts_3d.euler_angles_to_rotation_matrix(angle)
                # trans = np.matmul(rot_mat,
                #         car_pose[4:].transpose()).flatten()
                # pose_new = np.hstack([car_pose[1:4], trans])
                # pdb.set_trace()

                depth, mask = self.render_car(pose_new, car_name)
                total_mask, total_depth, _, is_valid = \
                         eval_uts.merge_inst(
                            {'depth': depth, 'mask': mask}, i + 1,
                            total_mask,
                            total_depth,
                            thresh=0)
                vis_rate[i] = np.float32(np.sum(mask))
                # if car_pose[-1] < 0:

            uts.plot_images({'image': image, 'depth': total_depth,
                'total_mask': total_mask})

            # mask = np.uint16(mask)
            # cv2.imwrite(car_params['car_mask_path'] + image_name + '.png',
            #         total_mask)
            # total_depth = np.uint16(total_depth * 256)
            # cv2.imwrite(car_params['car_depth_path'] + image_name + '.png',
            #         total_depth)

            ### Following is for resaving pose
            car_pose_json = []
            for i, car_pose in enumerate(car_poses):
                car_pose_dict = {}
                cur_area = np.sum(total_mask == (i + 1))
                area = np.round(
                    np.float32(cur_area) / (self.scale ** 2))
                vis_rate[i] = cur_area / (vis_rate[i] + np.spacing(1))
                depth = car_pose[-1]
                if area == 0:
                    print i, 'rendered mask has 0 area'
                    continue

                car_pose_dict['car_id'] = int(car_pose[0])
                car_pose_dict['pose'] = car_pose[1:].tolist()
                car_pose_dict['area'] = int(area)
                car_pose_dict['visible_rate'] = float(vis_rate[i])
                car_pose_json.append(car_pose_dict)

            with open(pose_file_out, 'w') as f:
                json.dump(car_pose_json, f, sort_keys=True, indent=4,
                    ensure_ascii=False)

    def resave_label(self, is_sim=False):
        """ re-save labelled car, or simulate multiple detection
        """
        import json
        for i, image_name in enumerate(self.image_list):
            if i % 10 == 1:
                print i, '/', len(self.image_list)
            cam_name = 'Camera_5'
            # for name in self.data_params['intrinsic'].keys():
            #     if name in image_name:
            #         cam_name = name
            #         break

            self.intrinsic = uts_3d.intrinsic_vec_to_mat(
                        self.data_params['intrinsic'][cam_name],
                        self.image_size)
            self.focal_len = self.intrinsic[0, 0]

            pose_file = self.data_params['car_pose_path'] + \
                    image_name[:-4] + '.poses'
            car_poses = data_libs[self.d_name].read_carpose(
                    pose_file, is_euler=True)
            if is_sim:
                car_poses = self.random_sample(car_poses)

            total_depth = 10000. * np.ones(self.image_size)
            total_mask = np.zeros(self.image_size)

            vis_rate = np.zeros(len(car_poses))
            for i, car_pose in enumerate(car_poses):
                car_pose['car_id'] = self.car_model.keys().index(car_pose['name'])
                if is_sim:
                    num = np.random.randint(6, size=1)[0] - 3
                    car_num = len(self.car_model) - 1
                    car_pose['car_id'] = min(max(car_pose['car_id'] + num, 0),
                            car_num)

                car_pose['name'] = self.car_model.keys()[car_pose['car_id']]
                depth, mask = self.render_car(
                         car_pose['pose'], car_pose['name'])

                total_mask, total_depth, _, is_valid = \
                         eval_uts.merge_inst(
                            {'depth': depth, 'mask': mask}, i + 1,
                            total_mask,
                            total_depth,
                            thresh=0)
                level = np.float32(np.sum(total_mask == (i + 1))) / (np.float32(np.sum(mask)) + np.spacing(1))
                vis_rate[i] = level

            # uts.plot_images({'total_mask': total_mask, 'depth': total_depth})
            keep_idx = []
            for i, car_pose in enumerate(car_poses):
                area = np.round(
                     np.float32(np.sum(total_mask == (i + 1))) / (self.scale ** 2))
                if area == 0:
                    continue

                if car_pose['name'] in self.car_model.keys():
                    car_pose['pose'] = car_pose['pose'].tolist()
                    car_pose['area'] = int(area)
                    car_pose['visible_rate'] = float(vis_rate[i])
                    car_pose.pop('name', None)
                    keep_idx.append(i)
                else:
                    raise Exception('%s not exist' % car_pose['name'])

            if len(keep_idx) < len(car_poses):
                car_poses = [car_poses[idx] for idx in keep_idx]

            save_path = self.data_params['car_pose_path_sim'] if is_sim \
                    else self.data_params['car_pose_path_new']
            pose_file_out = save_path + image_name[:-4] + '.json'
            with open(pose_file_out, 'w') as f:
                json.dump(car_poses, f, sort_keys=True, indent=4,
                    ensure_ascii=False)


    def find_label_subset(self):
        car_names = set()
        image_list = [image_name.strip() for image_name in open(
            self.data_params['train_list'])]
        image_list += [image_name.strip() for image_name in open(
            self.data_params['val_list'])]

        for i, image_name in enumerate(image_list):
            if i % 100 == 1:
                print i, '/', len(image_list)
            pose_file = self.data_params['car_pose_path'] + \
                    image_name[:-4] + '.poses'
            car_poses = data_libs[self.d_name].read_carpose(
                    pose_file, is_euler=True)
            for i, car_pose in enumerate(car_poses):
                car_name = car_pose['name']
                car_names.add((car_name, self.car_model[car_name]['car_type']))

        print len(car_names)
        # self.resave_car_model(data_ext='off', car_set=car_names)
        # self.resave_car_model(data_ext='pkl', car_set=car_names)

        self.compute_reproj_sim(list(car_names), 'sim_mat.txt')


    def compute_reproj_sim(self, car_names=None, out_file=None):

        self.intrinsic = uts_3d.intrinsic_vec_to_mat(
                    self.data_params['intrinsic']['Camera_5'],
                    self.image_size)

        if out_file is None:
            out_file = './sim_mat.txt'

        sim_mat = np.eye(len(self.car_model))
        for i in range(len(car_names)):
            for j in range(i, len(car_names)):
                if i == j:
                    continue

                name1 = car_names[i][0]
                name2 = car_names[j][0]
                ind_i = self.car_model.keys().index(name1)
                ind_j = self.car_model.keys().index(name2)
                sim_mat[ind_i, ind_j] = self.compute_reproj(name1, name2)
                sim_mat[ind_j, ind_i] = sim_mat[ind_i, ind_j]
                print name1, name2, sim_mat[ind_i, ind_j]

        np.savetxt(out_file, sim_mat, fmt='%1.6f')


    def compute_reproj(self, car_name1, car_name2):
        sims = np.zeros(10)
        for i, rot in enumerate(np.linspace(0, np.pi, num=10)):
            pose = np.array([0, rot, 0, 0, 0,5.5])
            depth1, mask1 = self.render_car(pose, car_name1)
            depth2, mask2 = self.render_car(pose, car_name2)
            # if car_name2 == '037-CAR02':
            #     uts.plot_images({'mask1': mask1, 'mask2': mask2})
            sims[i] = eval_uts.IOU(mask1, mask2)

        return np.mean(sims)


    def resave_car_model(self, data_ext='pkl', car_set=None):
        car_set = self.data_params['car_names'] if car_set is None else car_set
        if data_ext == 'pkl':
            import pickle
            for name, car_type in self.data_params['car_names']:
                car_file = '%s/%s.pkl' % (self.data_params['car_model_path_pkl'], name)
                with open(car_file, 'w') as f:
                    pickle.dump(self.car_model[name], f)
        else:
            for name, car_type in car_set:
                car_model = self.car_model[name]
                car_file = '%s/%s.off' % (self.data_params['car_model_path_off'], name)
                with open(car_file, 'w') as f:
                    f.write('OFF\n')
                    f.write('%d %d 0\n' % (car_model['vertices'].shape[0],\
                            car_model['faces'].shape[0]))
                    v = car_model['vertices']
                    for i in range(car_model['vertices'].shape[0]):
                        f.write('%.4f %.4f %.4f\n' % (v[i, 0], v[i, 1], v[i, 2]))
                    face = car_model['faces'] - 1
                    for i in range(car_model['faces'].shape[0]):
                        f.write('3 %d %d %d\n' % (face[i, 0], face[i, 1], face[i, 2]))


def mp_resave_new_label(split, part,  nproc):
    car_params = data_libs['apollo'].set_params_car(split)
    image_list = [image_name[:-4] for image_name in \
            sorted(os.listdir(car_params['car_pose_path']))]
    image_lists = uts.split_list(image_list, nproc)
    image_list = image_lists[part]
    params = data_libs['apollo'].set_params_disp(stereo_rect=True)
    env = Env(config, params, split=split)
    env.resave_new_label(image_list)


def resave_new_label(split='train', nproc=4):
    import multiprocessing as mp
    procs = []

    for i in range(nproc):
        # change to multi process sampling, initialize many dummy env & agent
        method = mp_resave_new_label
        arguments = (split, i, nproc)
        proc = mp.Process(target=method, args=arguments)
        proc.daemon = True
        proc.start()
        procs.append(proc)

    # wait all the procs are finished
    for proc in procs:
        proc.join()


def resave_name(data='apollo', split='train'):
    car_params = data_libs[data].set_params_car(split)

    # test_file_name = '171206_034630104_Camera_5'
    image_list = [image_name[:-4] for image_name in \
        sorted(os.listdir(car_params['car_pose_path']))]

    for i, image_name in enumerate(image_list):
        save_path = car_params['car_pose_path_json']
        pose_file_out = save_path + image_name[:-4] + '.json'
        pose_save_file = save_path[:-1] + '_new/' + image_name + '.json'
        os.system('cp %s %s' % (pose_file_out, pose_save_file))


if __name__ == '__main__':
    import config.policy_config_apollo as config
    config.scale = 1
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # filename = '%06d_10' % 1
    mp_resave_new_label('test', 0, 1)
    # resave_new_label(split='train')
    # resave_name()
    # params = data_libs['apollo'].set_params_disp(stereo_rect=True)
    # env = Env(config, params, split='test')
    # env.resave_new_label()
    # env.check_resave_new_label()

    # env.set_stat_plot()
    # env.find_label_subset()
    # env.compute_reproj_sim(params['car_names'], './sim_mat_79.txt')
    # env.resave_car_model(data_ext='pkl')

    # for set_name in ['train', 'val']:
    #     env = Env(config, params, split=set_name)
    #     # env.resave_label(is_sim=True)
    #     env.simulate_noisy_pose()
    # env.resave_images()

    # env.save_rect_image_set()
    # eval with inital pose
    # total_diff = []
    # total_iou = []
    # counter = 0
    # while True:
    #     env.get_image()
        # state, act = env.sample_state_rect()
        # print state['pose'], act['del_pose']
        # cur_act = {'del_pose': np.zeros((1, 6))}
        # init_act = env.init_pose(is_rand=True, state=state)
        # cur_act = {'del_pose': init_act - state['pose']}

        # print init_act, cur_act
        # reward, _, _, ins = env.step(cur_act, state, is_plot=True)
        # break

    # while True:
    #     state, act = env.sample_state_rect()
    #     total_diff.append(np.linalg.norm(state['pose'][0, 3:] - act['del_pose'][0, 3:]))
    #     cur_act = {'del_pose': np.zeros((1, 6))}
    #     reward, _, _, ins = env.step(cur_act)
    #     total_iou.append(np.max(ins['IOU']))
    #     if env.inst_counter == len(env.valid_id) and \
    #             env.counter == env.image_num:
    #         break
    #     counter += 1
    #     print counter

    # total_diff = np.array(total_diff)
    # total_iou = np.array(total_iou)
    # print '%.4f, %.4f, %.4f, %.4f' % (np.mean(total_diff), np.median(total_diff), np.mean(total_iou), np.median(total_iou))

    #     env.counter += 1
    #     logging.info('%d/ %d' % (env.counter, env.image_num))
    #     if env.counter == env.image_num:
    #         raise ValueError('All images are finished')
    # env = Env(config, params, split='minitrain')
    # while True:
    #     state = env.vis_apollo_render_depth()

    # env = Env(config, params, split='train')
    # while True:
    #     state = env.sample_state_rect()
    #     act = {'del_pose': np.zeros(6)}
    #     reward, _, _, ins = env.step(act, is_plot=True)
    #     logging.info('reward %s iou %s, delta %s' % (reward, ins['IOU'], ins['delta']))




