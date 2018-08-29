
import sys
sys.path.insert(0, "../")

import numpy as np
import cv2
import cython_util as cut
import utils.utils as uts
from collections import OrderedDict



def test_project():
    project = np.load('./test.npy')
    shape = [192, 256]
    depth, id_map = cut.gen_depth_map(project.astype(np.float32),
      shape[0], shape[1], 1)

    print depth

def test_extend_building():
    image = np.zeros((10, 10), dtype=np.int32)
    building_id = 1
    sky_id = 2
    image[0, 0] = 1
    image[2, 0] = 2
    image[9, 1] = 1
    image_o = cv2.imread('/home/peng/Data/zpark/Label/Record001/Camera_1/170427_222949577_Camera_1.png', cv2.IMREAD_UNCHANGED)
    # image_o = cv2.resize(image, (100, 100), interpolation=cv2.INTER_NEAREST)
    print image
    building_id = 25
    sky_id = 1
    image = cut.extend_building(np.int32(image_o), building_id, sky_id)
    print image
    uts.plot_images({'image_o':image_o, 'image_1':image})


def test_min_val():
    arr = np.float32(np.random.rand(10))
    min_val = cut.min_value_np(arr)
    print arr
    print np.min(arr), min_val


def test():
    depth = np.ones((100, 100), dtype=np.float32)
    normal = np.zeros((3, 100, 100), dtype=np.float32)
    intrinsice = np.array([1, 1, 50, 50], dtype=np.float32)

    normal = cut.depth2normals_np(depth, intrinsice)

    print normal.shape
    print normal.transpose((1, 2, 0))[:10, :10, :3]

def test_img():
    # depth = np.load('/home/peng/Data/visualization.npy')
    depth = cv2.imread('/home/peng/Data/kitti/000000_10.png')
    print np.amax(depth)
    mask = depth[:, :, 0] == 0
    depth = np.float32(1.0 / depth[:, :, 0]) * 1000
    depth[mask] = 0.0

    # depth = np.float32(1. / depth[:, :, 0])
    height, width = depth.shape
    # depth = cv2.resize(depth, (width / 3, height / 3))
    # intrinsic = np.array([1, 1, width / 2, height / 2], dtype=np.float32)
    intrinsic = np.array([959.0/width, 957.0/height, 696.0/width, 224.0/height],\
     dtype=np.float32)

    normal = cut.depth2normals_np(depth, intrinsic)
    normal = normal.transpose([1, 2, 0])
    normal[:, :, [1,2]] *= -1

    # uts.plot_images(OrderedDict([('depth', depth),
    #                              ('normal', (normal + 1.0)/2.)]),
    #                 layout=[2,1])
    uts.plot_images(OrderedDict([('depth', depth),
                                 ('normal', normal)]),
                    layout=[2,1])


if __name__ == '__main__':
    # test_img()
    # test_project()
    # test_min_val()
    test_extend_building()
