
import sys
sys.path.insert(0, "../")

import numpy as np
import numpy.testing as npt
import cython_util as cut
import utils.utils as uts
from collections import OrderedDict

def test():
    depth = np.ones((100, 100), dtype=np.float32)
    normal = np.zeros((3, 100, 100), dtype=np.float32)
    intrinsice = np.array([1, 1, 50, 50], dtype=np.float32)

    normal = cut.depth2normals_np(depth, intrinsice)

    print normal.shape
    print normal.transpose((1, 2, 0))[:10, :10, :3]


def test_img():
    depth = np.load('/home/peng/Data/visualization.npy')
    depth = np.float32(1. / depth[:, :, 0])
    height, width = depth.shape
    intrinsic = np.array([1, 1, width / 2, height / 2], dtype=np.float32)

    normal = cut.depth2normals_np(depth, intrinsic)
    normal = normal.transpose([1, 2, 0])
    print normal
    # uts.plot_images(OrderedDict([('depth', depth),
    #                              ('normal', (normal + 1.0)/2.)]),
    #                 layout=[4,2])


if __name__ == '__main__':
    test_img()
