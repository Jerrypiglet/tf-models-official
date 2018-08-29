# this functions preprocess the sun images ( including generating training image pairs,
# optical flow ground truth between them, the optical flow confidence )

import sys
sys.path.insert(0, "/home/peng/Paddle/python")
sys.path.insert(0, "../")

import os
import gflags

import cPickle as pkl
import numpy as np
# import pyximport
# pyximport.install(setup_args={'include_dirs': np.get_include()})

import cython_util as cuts
from collections import OrderedDict
import paddle.utils as pd_util
from paddle.utils import preprocess_util
import utils.utils as uts
import sun3d

gflags.DEFINE_integer('start', 0, 'Start id of the sun3d scenes')
gflags.DEFINE_integer('end', 100, 'End id of the sun3d scenes')

FLAGS = gflags.FLAGS

HOME='/home/peng/'
DATA_PATH=HOME+'/Data/sun3d/'
FLOW_PATH=HOME+'/Data/sun3d/flow/'


def find_image_depth_matching(scene):
    pairs = uts.read_file(DATA_PATH + scene + '/id_img2depth.txt')
    id_img2depth = {}
    for pair in pairs:
        image_name, depth_name = pair.split(' ')
        id_img2depth[image_name] = depth_name
    return id_img2depth


def gen_img_pair_data(scene, pair_num, id_img2depth):
    # for each scene, for each image, gen pair of images
    K = np.loadtxt(DATA_PATH + scene + '/intrinsics.txt')
    extrinsic_file = pd_util.preprocess_util.list_files(DATA_PATH + scene + '/extrinsics/')
    extrinsic_file.sort()
    extrinsic = np.reshape(np.loadtxt(extrinsic_file[-1]), (-1, 3, 4))
    # keep the original
    id_img2depth = OrderedDict(sorted(id_img2depth.items(), key=lambda t:t[0]))
    image_names = id_img2depth.keys()

    for i in range(0, len(image_names) - 30, 10):
        pair_id = np.random.choice(range(10, 30), 10, replace=False)
        for j in pair_id:
            image_path1 = DATA_PATH + scene + '/image/' + image_names[i] + '.jpg'
            image_path2 = DATA_PATH + scene + '/image/' + image_names[i + j] + '.jpg'
            depth_path1 = DATA_PATH + scene + '/depth/' + id_img2depth[image_names[i]] + '.png'
            depth_path2 = DATA_PATH + scene + '/depth/' + id_img2depth[image_names[i + j]] + '.png'

            # try:
            image1 = np.array(uts.load_image(image_path1))
            image2 = np.array(uts.load_image(image_path2))
            depth1 = uts.read_depth(depth_path1)
            depth2 = uts.read_depth(depth_path2)
            # except:
            #     continue

            print "image1 name: {}, image2 name: {} \
                   depth1 name: {}, depth2 name: {}".format(
                   image_names[i], image_names[i + j],
                   id_img2depth[image_names[i]], id_img2depth[image_names[i + j]])

            flow, is_valid = get_opt_flow(depth1, depth2, K,
                                          extrinsic[i, :, :],
                                          extrinsic[i + j, :, :],
                                          True, image1, image2)
            is_valid = False
            uts.plot_images(OrderedDict([('image1',image1),
                                         ('image2',image2),
                                         ('flowu',flow[:,:,0]),
                                         ('flowv',flow[:,:,1])]))

            # print is_valid
            if is_valid:
                flow_file = FLOW_PATH + scene + '/flow/' + \
                            image_names[i] + '_' + image_names[i + j] + '.pkl'
                print 'saving ' + flow_file
                with open(flow_file, 'wb') as f:
                    pkl.dump(flow, f, -1)


def get_opt_flow(depth1,
                 depth2,
                 K,
                 extr1,
                 extr2,
                 get_isvalid=False,
                 image1=None,
                 image2=None):
    # using pair of images to get the optical flow between them
    # read intrinsic
    assert(depth1.shape[0] == depth2.shape[0] and
           depth1.shape[1] == depth2.shape[1])

    height, width = depth1.shape
    pix_num = height * width

    xyz_camera = depth2xyz_camera(depth1, K)
    valid = xyz_camera[:, 3] > 0
    xyz_camera = xyz_camera[valid, 0:3]
    xyz_world = transform_c2w(xyz_camera, extr1)
    xyz_camera2 = transform_w2c(xyz_world, extr2)
    project = np.dot(K, xyz_camera2.transpose())
    project[0:2, :] /= project[2, :]

    x, y = np.meshgrid(range(1, width + 1), range(1, height + 1))
    x = x.reshape(pix_num)[valid].astype(np.float32)
    y = y.reshape(pix_num)[valid].astype(np.float32)
    flowx = project[0, :] - x
    flowy = project[1, :] - y
    flow = np.zeros(2 * height * width, dtype=np.float32)
    flow[np.concatenate((valid, valid))] =  np.concatenate((flowx, flowy))

    proj_depth = cuts.gen_depth_map(project.astype(np.float32), height, width)

    # remove points that has bad depth
    # good3D = np.abs(proj_depth - np.float32(depth2)) < 0.05
    # valid = np.logical_and(valid, good3D.reshape(height * width))

    # remove outlier
    std_flowx = np.std(flow[0:pix_num][valid])
    std_flowy = np.std(flow[pix_num:][valid])
    med_flowx = np.median(flow[0:pix_num][valid])
    med_flowy = np.median(flow[pix_num:][valid])

    none_outlier = np.logical_and(np.logical_and(
                                  med_flowx - 3*std_flowx < flow[0:pix_num],
                                  flow[0:pix_num] < med_flowx + 3*std_flowx),
                                  np.logical_and(
                                  med_flowy - 3*std_flowy < flow[pix_num:],
                                  flow[pix_num:] < med_flowy + 3*std_flowy))
    valid = np.logical_and(valid, none_outlier)

    flow = flow.reshape((2, pix_num))
    flow *= valid.astype(np.float32)
    flow = np.transpose(flow.reshape((2, height, width)), (1, 2, 0))

    is_valid = True
    if get_isvalid:
        new_x = project[0, :]
        new_y = project[1, :]

        within_indice = np.logical_and(np.logical_and(0 < new_x, new_x <= width),
                                       np.logical_and(0 < new_y, new_y <= height))
        if np.sum(within_indice.astype(np.float32)) / pix_num < 0.5:
            is_valid = False

        ind2 = np.int32(new_y[within_indice]-1) * width + np.int32(new_x[within_indice]-1)
        ind1 = (y[within_indice]  -1) * width + (x[within_indice] - 1)

        image1 = image1.reshape((height*width, 3))
        image2 = image2.reshape((height * width, 3))
        err = np.sum(np.abs(image1[ind1.astype(np.int32), :].astype(np.float32)\
                            - image2[ind2, :].astype(np.float32)), axis=1)
        if np.sum(np.float32(err > 250))/np.float32(err.size) > 0.5:
            is_valid = False
    return flow, is_valid


def depth2xyz_camera(depth, K):
    depth = np.float32(depth)
    height, width = depth.shape
    x, y = np.meshgrid(range(1, width+1), range(1, height+1))
    xyz_camera = np.zeros((height, width, 4), dtype=np.float32)
    xyz_camera[:, :, 0] = (x.astype(np.float32()) - K[0, 2]) * depth / K[0, 0]
    xyz_camera[:, :, 1] = (y.astype(np.float32()) - K[1, 2]) * depth / K[1, 1]
    xyz_camera[:, :, 2] = depth
    xyz_camera[:, :, 3] = np.float32(depth > 0)
    xyz_camera = xyz_camera.reshape((-1, 4))

    return xyz_camera


def transform_c2w(xyz_camera, extr_c2w):
    xyz_world = np.dot(xyz_camera, extr_c2w[:, 0:3].transpose()) + \
             extr_c2w[:, 3].transpose()
    # k x 3
    return xyz_world


def transform_w2c(xyz_world, extr_c2w):
    xyz_proj = np.dot(np.linalg.inv(extr_c2w[:, 0:3]),
                      (xyz_world - extr_c2w[:, 3].transpose()).transpose())
    return xyz_proj.transpose()


def main(argv):
    argv=FLAGS(argv) # parse argv to FLAG
    scene_names = uts.read_file(DATA_PATH + 'test_sun3d.txt')
    scene_names = scene_names[FLAGS.start:FLAGS.end]

    for scene_name in scene_names:
        scene_name += '/'
        print FLOW_PATH + scene_name + 'flow/'
        # down load file
        if not os.path.exists(DATA_PATH + scene_name + 'id_img2depth.txt'):
            print 'Retrieve Data'
            os.system("/home/peng/SUN3DCppReader/src/build/SUN3DCppReader " \
                      + scene_name + " " + DATA_PATH)
            if not os.path.exists(DATA_PATH + scene_name + '/id_img2depth.txt'):
                continue

        if not os.path.exists(FLOW_PATH + scene_name + '/flow/'):
            uts.mkdir_if_need(FLOW_PATH + scene_name + '/flow/')
            id_img2depth = find_image_depth_matching(scene_name)
            gen_img_pair_data(scene_name, 500, id_img2depth)

def test(argv):
    params = sun3d.set_params()
    for scene_name in params['train_scene']:
        id_img2depth = find_image_depth_matching(scene_name)
        gen_img_pair_data(scene_name, 500, id_img2depth)


if __name__ == '__main__':
    # main(sys.argv)
    test(sys.argv)
