# preprocess the training images
import os
import cv2
import sys
import pdb

python_version = sys.version_info.major
if python_version == 2:
    import cPickle as pkl
    from paddle.utils import preprocess_util
    import cython_util as cut
    import utils.utils_3d as uts_3d
elif python_version == 3:
    import _pickle as pkl

import itertools
import numpy as np
import utils.utils as uts
import utils.utils_3d as uts_3d


__all__ = ['train', 'test']

HOME='/home/peng/Data/'
FLOW_PATH=HOME+'sun3d/flow/'
DATA_PATH=HOME+'sun3d/'


def set_params():
    params = {'stage': 2}
    params['data_path'] = '/home/peng/Data/sun3d/'
    params['output_path'] = '/home/peng/baidu/VideoSegRec/DeMoN/output/'
    params['flow_path'] = FLOW_PATH
    params['upsample_path'] = DATA_PATH + 'upsample/'

    scenes = os.listdir(params['flow_path'])
    params['scene_names'] = []
    for scene in scenes:
        sub_scenes = os.listdir(params['flow_path'] + scene)
        for sub_scene in sub_scenes:
            params['scene_names'].append(scene + '/' + sub_scene)

    # height width
    params['size'] = [192, 256]
    params['size_stage'] = [[6, 8], [48, 64]]
    params['batch_size'] = 32

    params['train_scene'] = params['scene_names'][0:50]
    params['test_scene'] = uts.read_file(params['data_path'] \
        + 'test_sun3d.txt')
    params['train_scene'] = uts.rm_b_from_a(params['train_scene'],
                                      params['test_scene'])

    # normalized intrinsic value
    params['intrinsic'] = np.array([0.89115971, 1.18821287, 0.5, 0.5])
    params['sample_rate'] = 0.05

    return params


def get_image_depth_matching(scene):
    pairs = uts.read_file(DATA_PATH + scene + '/id_img2depth.txt')
    id_img2depth = {}
    for pair in pairs:
        image_name, depth_name = pair.split(' ')
        id_img2depth[image_name] = depth_name
    return id_img2depth


def reader_creator(scene_names, height, width,
                   tasks=['flow', 'depth', 'normal'],
                   max_num=None):
    def reader():
        for i, scene_name in enumerate(scene_names):
            if ('depth' in tasks) or ('normal' in tasks):
                id_img2depth = get_image_depth_matching(scene_name)

            if 'trans' in tasks:
                extrinsic_file = preprocess_util.list_files(DATA_PATH + \
                 scene_name + '/extrinsics/')
                extrinsic_file.sort()
                extrinsic = np.reshape(np.loadtxt(extrinsic_file[-1]),
                    (-1, 3, 4))

            # intrinsic = np.array([0.89115971, 1.18821287, 0.5, 0.5])
            K = np.loadtxt(DATA_PATH + scene_name + '/intrinsics.txt')
            intrinsic = np.asarray([K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                                    dtype=np.float32)

            image_list = preprocess_util.list_files(FLOW_PATH + scene_name + '/flow/')
            prefix_len = len(FLOW_PATH + scene_name + '/flow/')
            image_num = len(image_list) if max_num is None \
                                        else min(len(image_list), max_num)
            for j in range(image_num):
                pair_name = image_list[j][prefix_len:-4]
                #print 'loading ' + pair_name
                image_name1, image_name2 = pair_name.split('_')
                image_path1 = DATA_PATH + scene_name + '/image/' + image_name1 + '.jpg'
                image_path2 = DATA_PATH + scene_name + '/image/' + image_name2 + '.jpg'
                flow_path = FLOW_PATH + scene_name + '/flow/' + pair_name + '.pkl'
                depth_path1 = DATA_PATH + scene_name + '/depth/' + \
                              id_img2depth[image_name1] + '.png'

                image1 = cv2.imread(image_path1)
                image2 = cv2.imread(image_path2)

                # normalize intrinsic
                intrinsic[[0, 2]] = intrinsic[[0, 2]] / image1.shape[1];
                intrinsic[[1, 3]] = intrinsic[[1, 3]] / image1.shape[0];

                image1 = cv2.resize(image1,
                    (width, height), interpolation=cv2.INTER_LINEAR)
                image2 = cv2.resize(image2,
                    (width, height), interpolation=cv2.INTER_LINEAR)
                image1 = uts.transform(image1, height, width)
                image2 = uts.transform(image2, height, width)


                weight = np.ones((height, width)) > 0
                outputs_task=[]
                if 'flow' in tasks:
                    with open(flow_path, 'rb') as flow_file:
                        flow = pkl.load(flow_file)

                    flow = cv2.resize(flow, (width, height),
                            interpolation=cv2.INTER_NEAREST)
                    flow[:, :, 0] = flow[:, :, 0] / width
                    flow[:, :, 1] = flow[:, :, 1] / height
                    flow = flow.transpose((2, 0, 1)).flatten()
                    outputs_task.append(flow)


                if 'trans' in tasks:
                    # compute camera 2 to camera 1 motion vector
                    id1 = int(image_name1.split('-')[0])
                    id2 = int(image_name2.split('-')[0])
                    r1 = extrinsic[id1, :, :3]
                    r2 = extrinsic[id2, :, :3]
                    t1 = extrinsic[id1, :, 3]
                    t2 = extrinsic[id2, :, 3]
                    r_rel = uts.rotationMatrixToEulerAngles(
                        np.matmul(r2.transpose(), r1))
                    t_rel = np.matmul(r2.transpose(), t1 - t2)
                    outputs_task.append(r_rel)
                    outputs_task.append(t_rel)


                if 'depth' in tasks:
                    depth1 = uts.read_depth(depth_path1)
                    depth1 = cv2.resize(depth1, (width, height),
                            interpolation=cv2.INTER_NEAREST)
                    depth1_inv = depth1.copy()
                    depth1_inv[depth1 > 0] = 1 / depth1[depth1 > 0]
                    weight = np.logical_and(weight, depth1_inv > 0)

                    # pdb.set_trace()
                    depth1_inv = depth1_inv.flatten()
                    assert not np.any(np.logical_or(np.isinf(depth1_inv),
                                                    np.isnan(depth1_inv)))
                    outputs_task.append(depth1_inv)


                if 'normal' in tasks:
                    if not ('depth1' in locals()):
                        depth1 = uts.read_depth(depth_path1)
                        depth1 = cv2.resize(depth1, (width, height),
                                interpolation=cv2.INTER_NEAREST)
                    normal1 = cut.depth2normals_np(depth1, intrinsic)
                    normal = normal1.transpose((2, 0, 1)).flatten()
                    outputs_task.append(normal)

                weight = np.float32(weight).flatten()
                outputs = [image1, image2, weight, intrinsic]
                outputs = outputs + outputs_task
                outputs = tuple(outputs)

                yield outputs

    return reader


def load_image_pair(scene_name, pair_name, id_img2depth=None,
                    get_normal=True):
    prefix_len = len(FLOW_PATH + scene_name + '/flow/')
    pair_name = pair_name[prefix_len:-4]
    K = np.loadtxt(DATA_PATH + scene_name + '/intrinsics.txt')
    intrinsic = np.asarray([K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                            dtype=np.float32)

    #print 'loading ' + pair_name
    image_name1, image_name2 = pair_name.split('_')
    image_path1 = DATA_PATH + scene_name + '/image/' + image_name1 + '.jpg'
    image_path2 = DATA_PATH + scene_name + '/image/' + image_name2 + '.jpg'
    flow_path = FLOW_PATH + scene_name + '/flow/' + pair_name + '.pkl'

    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    height, width, channel = image1.shape

    with open(flow_path, 'rb') as flow_file:
        flow = pkl.load(flow_file)
    if id_img2depth == None:
        return image1, image2, flow
    else:
        depth_path1 = DATA_PATH + scene_name + '/depth/' + \
                      id_img2depth[image_name1] + '.png'
        intrinsic[[0, 2]] = intrinsic[[0, 2]] / width;
        intrinsic[[1, 3]] = intrinsic[[1, 3]] / height;

        depth1 = uts.read_depth(depth_path1)
        if get_normal:
            normal1 = cut.depth2normals_np(depth1, intrinsic)
            normal1 = normal1.transpose((1,2,0))
            return image1, image2, flow, depth1, normal1
        else:
            return image1, image2, flow, depth1


def train(scene_names, height, width, tasks):
    return reader_creator(scene_names, height, width, tasks)


def test(scene_names, height, width, tasks):
    return reader_creator(scene_names, height, width, tasks, 80)


# def reader_creator_upsampler(scene_names,  upsampler, rate,
#                              height, width, max_num=None):
#     def reader():
#         intrinsic = np.array([0.89115971, 1.18821287, 0.5, 0.5])
#         for i, scene_name in enumerate(scene_names):
#             id_img2depth = get_image_depth_matching(scene_name)
#             image_list = preprocess_util.list_files(FLOW_PATH + scene_name + '/flow/')
#             prefix_len = len(FLOW_PATH + scene_name + '/flow/')
#             image_num = len(image_list) if max_num is None \
#                                         else min(len(image_list), max_num)
#             image_id = np.random.randint(len(image_list),
#                                          size=image_num)
#             for j in image_id:
#                 pair_name = image_list[j][prefix_len:-4]
#                 #print 'loading ' + pair_name
#                 image_name1, image_name2 = pair_name.split('_')
#                 image_path1 = DATA_PATH + scene_name + '/image/' + image_name1 + '.jpg'
#                 image_path2 = DATA_PATH + scene_name + '/image/' + image_name2 + '.jpg'
#                 depth_path1 = DATA_PATH + scene_name + '/depth/' + \
#                               id_img2depth[image_name1] + '.png'
#                 image1 = cv2.imread(image_path1)
#                 image2 = cv2.imread(image_path2)

#                 depth1 = uts.read_depth(depth_path1)
#                 depth1 = cv2.resize(depth1, (width, height),
#                         interpolation=cv2.INTER_NEAREST)
#                 depth1_inv = depth1.copy()
#                 depth1_inv[depth1 > 0] = 1. / depth1[depth1 > 0]
#                 weight = np.float32(depth1_inv > 0)
#                 depth_gt_down = uts_3d.down_sample_depth(depth1,
#                                              method='uniform',
#                                              percent=rate,
#                                              K=intrinsic)

#                 depth_net = upsampler.demon_net_depth(depth_gt_down,
#                         [image1, image2])

#                 depth_net_inv = depth_net.copy()
#                 depth_net_inv[depth_net  > 0] = 1. / depth_net[depth_net > 0]

#                 # pdb.set_trace()
#                 depth1_inv = depth1_inv.flatten()
#                 assert not np.any(np.logical_or(np.isinf(depth1_inv),
#                                                 np.isnan(depth1_inv)))
#                 weight = np.float32(weight).flatten()
#                 outputs = [image1, depth_net_inv, depth1_inv,  weight]
#                 outputs = tuple(outputs)

#                 yield outputs

#     return reader


# def train_upsampler(scene_names, upsampler, rate, height, width, max_num=10):
#     return reader_creator_upsampler(scene_names,
#             upsampler, rate, height, width, max_num)


# def test_upsampler(scene_names, upsampler, rate, height, width, max_num=50):
#     return reader_creator_upsampler(scene_names, upsampler, rate,
#             height, width, max_num)


def reader_creator_upsampler(scene_names, rate,
                             height, width, max_num=None,
                             is_inverse=False):
    def reader():
        intrinsic = np.array([0.89115971, 1.18821287, 0.5, 0.5])

        for i, scene_name in enumerate(scene_names):
            id_img2depth = get_image_depth_matching(scene_name)
            upsample_output_path = FLOW_PATH + scene_name + \
                                   '/pair_depth/' + str(rate) + '/'

            if not os.path.exists(upsample_output_path):
                continue

            image_list = preprocess_util.list_files(upsample_output_path)
            prefix_len = len(upsample_output_path)
            image_num = len(image_list) if max_num is None \
                                        else min(len(image_list), max_num)
            image_id = np.random.randint(len(image_list), size=image_num) \
                       if image_num < len(image_list) else range(len(image_list))

            for j in image_id:
                pair_name = image_list[j][prefix_len:-4]
                #print 'loading ' + pair_name
                image_name1, _ = pair_name.split('_')
                image_path1 = DATA_PATH + scene_name + '/image/' + image_name1 + '.jpg'
                depth_path1 = DATA_PATH + scene_name + '/depth/' + \
                              id_img2depth[image_name1] + '.png'

                depth_net = np.load(image_list[j])
                image1 = cv2.imread(image_path1)
                image1 = uts.transform(image1, height, width)

                depth1 = uts.read_depth(depth_path1)
                depth1 = cv2.resize(depth1, (width, height),
                        interpolation=cv2.INTER_NEAREST)
                weight = np.float32(depth1 > 0)

                if is_inverse:
                    depth1= uts_3d.inverse_depth(depth1)
                    depth_net= uts_3d.inverse_depth(depth_net)

                weight = weight.flatten()
                depth1 = depth1.flatten()
                depth_net = depth_net.flatten()

                assert not np.any(np.logical_or(np.isinf(depth1),
                                                np.isnan(depth1)))
                weight = np.float32(weight).flatten()
                outputs = [image1, depth_net, depth1,  weight]
                outputs = tuple(outputs)
                yield outputs

    return reader


def train_upsampler(scene_names, rate, height, width, max_num=None,
                    is_inverse=False):
    return reader_creator_upsampler(
        scene_names, rate, height, width, max_num, is_inverse)


def test_upsampler(scene_names, rate, height, width, max_num=None,
                   is_inverse=False):
    return reader_creator_upsampler(
        scene_names, rate, height, width, max_num, is_inverse)

if __name__ == "__main__":
    test()
