import os
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
import data.kitti as kitti
import data.xroad as xroad
import data.data_setting as data_setting
from collections import OrderedDict
import logging
import pdb

DTYPE = np.float32
np.set_printoptions(threshold=np.nan)

data_libs = {}
data_libs['kitti'] = kitti
data_libs['xroad'] = xroad


class Env(object):
    def __init__(self, c, data_params, data='kitti', split='train'):

        self.data_params = data_params
        self.config = c
        self.d_name = data
        num_range = range(1, 194) if split == 'train' else range(1, 20)
        # num_range = range(1, 5) if split == 'train' else range(1, 5)
        self.image_list = ['%06d_10' % i for i in num_range]

        # random perturb later
        self.list_order = range(len(self.image_list))
        self.image_num = len(self.image_list)
        self.counter = 0
        self.inst_counter = 0
        self.inst_counter_g = 0
        self.timestep_limit = c.timestep_limit

        if data == 'kitti':
            self.floor_plane = 1.0
            self.base = 0.54

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
        for car_name in data_params['car_names']:
            car_file = '%s/%s.mat' % (data_params['car_model_path'], \
                        car_name)
            self.car_model[car_name] = eval(data + '.load_car(car_file)')

        self.state_names = ['image', 'depth', 'mask', 'pose', 'render_depth']
        self.action_names = ['del_pose']
        self.reward_names = ['reward']

        self.state_update_names = ['pose'] # change during the steps
        self.inspector_names = ['IOU', 'delta']
        self.state = OrderedDict({})
        if self.config.is_crop:
            self.setting = data_setting.get_policy_data_setting(self)


    def _round_to_even(self, num):
        return np.ceil(num / 4.) * 4.


    def get_image(self):
        # sample an image
        if self.counter == self.image_num:
            self.counter = 0
            # logging.info('inst number %d' % self.inst_counter_g)
            self.inst_counter_g = 0
            self.list_order = np.random.permutation(self.image_num)

        idx = self.list_order[self.counter]
        image_name = self.image_list[idx]

        self.image_name = image_name
        if os.path.exists(self.data_params['car_inst_path']):
            label = cv2.imread(self.data_params['car_inst_path'] + \
                    image_name + '.png', cv2.IMREAD_UNCHANGED)
            label = cv2.resize(label, tuple(self.image_size[::-1]),
                    interpolation=cv2.INTER_NEAREST)
            self.masks = uts.label2mask(label)
        else:
            self.masks = kitti.get_instance_masks(self.data_params,
                    image_name, 'car', sz=self.image_size)

        if len(self.masks) == 0:
            return False

        self.image = cv2.imread(self.data_params['image_path'] \
                           + image_name + '.png', cv2.IMREAD_UNCHANGED)
        self.depth = kitti.depth_read(self.data_params['cnn_disp_path'] \
                           + image_name + '.png')

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


    def render_car(self, pose, car_id):
        car_name, car = self.car_model.items()[car_id]
        vert = uts_3d.project(pose, car['scale'], car['vertices'])
        intrinsic = np.float64(
                uts_3d.intrinsic_mat_to_vec(self.intrinsic))
        depth, mask = render.renderMesh_np(
                np.float64(vert),
                np.float64(car['faces']),
                intrinsic, self.image_size[0], self.image_size[1])

        return depth, mask


    def sample_state(self):
        if self.counter == 0:
            while not self.get_image():
                self.counter += 1
                if self.counter == self.image_num:
                    raise ValueError('All images has no mask')
            self.counter += 1

        if self.inst_counter == len(self.masks):
            while not self.get_image():
                self.counter += 1
            self.counter += 1

        if self.inst_counter == 0:
            # change to network input
            self.state['image'] = self.image.copy()
            self.state['depth'] = self.depth.copy()

        self.state['mask'] = self.masks[self.inst_counter].copy()

        self._centroid_pose = self.init_pose()
        depth, mask = self.render_car(self._centroid_pose[0], 0)
        self.state['render_depth'] = depth
        self.state['pose'] = DTYPE(self._centroid_pose.copy())
        self.state['init_pose'] = DTYPE(self._centroid_pose.copy())

        self.inst_counter += 1
        self.inst_counter_g += 1

        return self.state

    def depth_convert(self, depth, masks, valid_ids):

        res_depth = depth.copy()
        median_depths = np.zeros(len(valid_ids))

        for i, ids in enumerate(valid_ids):
            mask_sm = masks == ids
            pixs = np.where(mask_sm)
            pixs = np.vstack(pixs)
            depths= depth[pixs[0, :], pixs[1, :]]
            median_depths[i] = np.sort(depths)[len(depths) // 2]
            res_depth[mask_sm] = median_depths[i]

        return res_depth, median_depths


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

        self.masks = displets['mask']
        self.poses = np.zeros((inst_num, 6))
        self.poses[:, [1, 3, 4, 5]] = displets['poses']
        self.poses[:, 4] = -1 * (self.poses[:, 4] + 1.6)

        self.valid_id = np.unique(self.masks)
        self.valid_id = self.valid_id[self.valid_id > 0]

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

            if self.config.reward_name == 'mask_err':
                self.layer_depth, self.median_depths = self.depth_convert(
                        self.depth, self.masks, self.valid_id)

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

        if self.inst_counter == len(self.valid_id):
            while not self.get_mcmc_image():
                self.counter += 1
            self.counter += 1

        self.state['image'] = self.image.copy()
        self.state['depth'] = self.depth.copy()
        car_inst_id = self.valid_id[self.inst_counter]
        self.state['image_name'] = self.image_name
        self.state['inst_id'] = car_inst_id
        self.state['mask'] = self.masks == car_inst_id
        self.state['intrinsic'] = self.intrinsic

        if self.config.reward_name == 'mask_err':
            self.state['layer_depth'] = self.layer_depth.copy()
            self.state['inst_median_depth'] = self.median_depths[self.inst_counter]

        # direct regression without consider
        pose = self.poses[car_inst_id - 1].copy()[None, :]
        self._centroid_pose = self.init_pose()
        self.state['pose'] = DTYPE(self._centroid_pose.copy())
        # depth, mask = self.render_car(self.state['pose'][0], 0)
        # self.state['render_depth'] = depth

        # action = {'del_pose': pose - self.state['pose']}
        action = {'del_pose': pose}
        # logging.info('%s' % mcmc_pose)
        # uts.plot_images({'mask': self.state['mask'],
        #                  'depth': self.state['depth']})
        if self.config.is_crop:
            crop = uts.get_mask_bounding_box(self.state['mask'])
            self.state['crop'] = crop
            image_names = ['render_depth', 'image', 'depth', 'mask']
            if self.config.reward_name == 'mask_err':
                image_names += ['layer_depth']

            for name in image_names:
                if name in self.state:
                    temp = uts.crop_image(self.state[name], crop)
                    inter = self.setting[name]['interpolation']
                    self.state[name], pad = uts.resize_and_pad(
                                np.float32(temp),
                                tuple([self.config.height, self.config.width]),
                                interpolation=inter,
                                get_pad_sz=True)
                    self.state['pad'] = pad
            # print self.state['crop'], crop
            # uts.plot_images({'render_depth': self.state['render_depth'],
            #                  'mask': self.state['mask'],
            #                  'depth': self.state['depth']})
        self.inst_counter += 1
        self.inst_counter_g += 1

        return self.state, action


    def pix2point(self, pix):

        z = self.depth[pix[0], pix[1]]
        x = np.float32([pix[1], pix[0], 1.0]) * z
        x = np.dot(np.linalg.inv(self.intrinsic), x)
        v = x / np.linalg.norm(x)
        x = x + v
        x[1] = -1.0 * (x[1] + self.floor_plane)
        return x


    def init_pose(self, is_rand=False, state=None):
        """ since there are lots of outlier at boundary, we do erosion of
            boundary for avoidance of far-pixels
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
            depths= self.depth[pixs[0, :], pixs[1, :]]
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

        # sample rotation from one of the 4th init
        if is_rand:
            idx = np.random.randint(4)
            pose[:, 1] = self.init_rot[idx]

        return pose


    def step(self, act, cur_state=None, get_res=False, is_plot=False):
        """ reward_name: what kind of reward to use
                         'mcmc': the reward function from CVPR 2015
            input state is mostly for solving multi process issue
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
            self.intrinsic = cur_state['intrinsic']
            self.focal_len = self.intrinsic[0, 0]

        if reward_name == 'mask_err':
            cur_layer_depth = self.state['layer_depth'] if cur_state is None \
                    else cur_state['layer_depth']
            cur_inst_median_depth = self.state['inst_median_depth'] if cur_state \
                    is None else cur_state['inst_median_depth']

        # logging.info('%s %s' % ('act', act['del_pose']))
        # logging.info('%s %s' % ('pose', self.state['pose']))
        # we don't consider rotation up and side, depth must > 0.1
        next_pose[:, 5] = max(0.5, next_pose[:, 5])
        max_reward = -1 * np.inf
        res = {}

        intrinsic = np.float64(uts_3d.intrinsic_mat_to_vec(self.intrinsic))
        for i, (car_name, car) in enumerate(self.car_model.items()):
            vert = uts_3d.project(next_pose[0],
                    car['scale'], car['vertices'])
            depth, mask = render.renderMesh_np(
                    np.float64(vert),
                    np.float64(car['faces']),
                    intrinsic, self.image_size[0], self.image_size[1])

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


            if reward_name == 'mine':
                IOU, delta, reward['reward'][0, i] = \
                        eval_uts.compute_reward(cur_mask, cur_depth,
                        mask, depth, self.occ_thr)
                IOUs[i] = IOU
                deltas[i] = delta
                # logging.info('%s, %s' %(IOU, delta))
                # if IOU <= 1e-9:
                #     if np.sum(mask) > 0:
                #         center1 = uts.get_mask_center(cur_mask) \
                #                 / self.image_size
                #         center2 = uts.get_mask_center(mask) \
                #                 / self.image_size
                #         reward['reward'][0, i] += np.exp(
                #             -30 * np.sum(np.square(center1 - center2)))

            elif reward_name == 'mcmc':
                cur_disp = uts_3d.depth2disp(cur_depth, self.focal_len, self.base)
                disp = uts_3d.depth2disp(depth, self.focal_len, self.base)
                IOU, delta, reward['reward'][0, i] = eval_uts.mcmc_reward(
                        cur_mask, cur_disp, mask, disp)
                IOUs[i] = IOU
                deltas[i] = delta

            elif reward_name == 'mask_err':
                cur_disp = uts_3d.depth2disp(cur_depth, self.focal_len, self.base)
                disp = uts_3d.depth2disp(depth, self.focal_len, self.base)
                cur_layer_disp = uts_3d.depth2disp(cur_layer_depth, self.focal_len,
                        self.base)
                inst_disp = self.focal_len * self.base / cur_inst_median_depth
                IOU, delta, reward['reward'][0, i] = eval_uts.mcmc_reward_v2(
                        cur_mask, cur_disp, cur_layer_disp, inst_disp, mask, disp)
                IOUs[i] = IOU
                deltas[i] = delta

            else:
                raise ValueError('no given reward %s' % reward_name)

            if get_res:
                if max_reward < reward['reward'][0, i]:
                    max_reward = reward['reward'][0, i]
                    res.update({'mask': mask, 'depth': depth})

            if IOU > 0.95 and delta > 0.9:
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

        # print reward
        if get_res:
            return reward, done, {'pose': next_pose}, inspector, res
        else:
            return reward, done, {'pose': next_pose}, inspector


if __name__ == '__main__':
    import config.policy_config as config
    config.is_crop = True

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info('%s' % config)
    filename = '%06d_10' % 1
    params = kitti.set_params_disp()
    env = Env(config, params)

    while True:
        state, act = env.sample_mcmc_state()
        act = {'del_pose': np.zeros(6)}
        reward, _, _, ins = env.step(act, cur_state=state, is_plot=True)
        # reward, _, _, ins = env.step(act, is_plot=True)
        logging.info('reward %s iou %s, delta %s' % (reward, ins['IOU'], ins['delta']))




