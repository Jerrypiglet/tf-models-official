import cv2
import numpy as np
import json

from collections import OrderedDict
import utils.transforms as trs
import utils.utils_3d as uts_3d


def get_policy_data_setting(env):
    transforms = OrderedDict({})
    rescale = lambda x: x

    transforms['image'] = {'transform': trs.image_transform,
                           'interpolation': cv2.INTER_CUBIC,
                            'is_image': True}

    transforms['depth'] = {'transform': trs.depth_transform,
                           'interpolation': cv2.INTER_NEAREST,
                           'is_image': True}

    transforms['render_depth'] = {'transform': trs.depth_transform,
                                  'interpolation': cv2.INTER_NEAREST,
                                  'is_image': True}
    transforms['layer_depth'] = transforms['render_depth']

    transforms['mask'] = {'transform': trs.mask_transform,
                          'interpolation': cv2.INTER_NEAREST,
                          'is_image': True}

    transforms['pose'] = {'transform': rescale,
                          'is_image': False}

    transforms['crop'] = {'transform': trs.crop_transform,
                          'is_image': False,
                          'params': {'size': env.image_size}}

    transforms['init_pose'] = {'transform': rescale,
                               'is_image': False, 'params': {}}

    return transforms


def get_crop_policy_data_setting(env):
    transforms = OrderedDict({})
    rescale = lambda x: x

    transforms['image'] = {'transform': trs.image_transform,
                           'interpolation': cv2.INTER_CUBIC,
                           'is_image': True}

    transforms['depth'] = {'transform': trs.depth_transform,
                           'interpolation': cv2.INTER_NEAREST,
                           'is_image': True}

    transforms['mask'] = {'transform': trs.mask_transform,
                           'interpolation': cv2.INTER_NEAREST,
                           'is_image': True}

    transforms['pose'] = {'transform': rescale,
                          'is_image': False}

    transforms['crop'] = {'transform': trs.crop_transform,
                           'is_image': False,
                           'params': {'size': env.image_size}}

    transforms['init_pose'] = {'transform': rescale,
                               'is_image': False, 'params': {}}

    return transforms


def padding(pose, pad_size):
    num = pose.shape[0]
    output = np.pad(pose, ((0, pad_size-num), (0, 0)),
         'constant')
    return output


def permute_pose(pose_in, pad_size=0):
    assert isinstance(pose_in, list)

    output = np.zeros((len(pose_in), 6))
    for i, car_pose in enumerate(pose_in):
        rot_i, trans_i = np.float32(car_pose['pose'][:3]), \
                np.float32(car_pose['pose'][3:])
        trans, rot = uts_3d.random_perturb(trans_i, rot_i,
                    3.0, np.pi / 6.)
        output[i] = np.concatenate([trans, rot]).flatten()

    if pad_size > 0:
        output = padding(output, pad_size)
    output = output[None, :, :]
    return output


def concat_pose(pose_in, pad_size=0):
    output = np.zeros((len(pose_in), 6))
    for i, car_pose in enumerate(pose_in):
        output[i] = np.float32(car_pose['pose'])

    output = padding(output, pad_size)
    output = output[None, :, :]
    return output


def json_loader(file_name):
    with open(file_name) as f:
        res = json.load(f)

    return res


def get_gnn_data_setting(params, config):
    data_setting = OrderedDict({})
    data_setting['pose_in'] = {'transform': permute_pose,
                       'reader': json_loader,
                       'path': params['car_pose_path_new'],
                       'ext': '.json',
                       'trans_params': {'pad_size': config.num_point}}

    label_setting = OrderedDict({})
    label_setting['pose_out'] = {'transform': concat_pose,
                       'reader': json_loader,
                       'path': params['car_pose_path_new'],
                       'ext': '.json',
                       'trans_params': {'pad_size': config.num_point}}

    return data_setting, label_setting


def data_transform_4_network(state, state_names, data_transform):
    state_data = OrderedDict({})

    # for putting into hashing table
    state_data['image_name'] = state['image_name']
    state_data['inst_id'] = state['inst_id']
    for name in state_names:
        trans = data_transform[name]['transform']
        params = data_transform[name]['params'] if 'params' in \
                data_transform[name].keys() else {}

        if 'depth' in name:
            params = {'mean_depth': state['pose'][0, -1]}
        state_data[name] = trans(state[name], **params)
    return state_data
