import sys
paddle_root = '/home/peng/Paddle/'
sys.path.insert(0, paddle_root + 'python')
sys.path.insert(0, paddle_root + 'paddle/')
sys.path.insert(1, '/home/peng/libigl/python/')
sys.path.insert(1, "../")

import pdb
import gflags
import cv2
import numpy as np

import data.sun3d as sun3d
import utils.utils as uts
import utils.utils_3d as uts_3d
import upsampler
from utils.vis import visualize_prediction
from paddle.utils import preprocess_util
from collections import OrderedDict
import matplotlib
from matplotlib import cm
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot

FLAGS = gflags.FLAGS

gflags.DEFINE_boolean('v', False, \
                      'whether plot the eval resutls')

def eval_depth(depth_in, depth_gt):
    num_image = len(depth_in)
    error_acc = 0.0
    num_acc = 0.0
    for i in range(num_image):
        mask = depth_gt[i] > 1e-6
        error = np.abs(depth_in[i] - depth_gt[i]) \
                / np.maximum(depth_gt[i], 1e-6)
        error = error * mask
        error_acc += np.sum(error)
        num_acc += np.sum(mask)

    return error_acc / num_acc


def test_all():
    pass


def test_single():
    with open('../test/depth_gt.npy', 'rb') as f:
        depth_gt = np.load(f)
    with open('../test/depth_res.npy', 'rb') as f:
        depth_res = np.load(f)
    vis = False
    params = sun3d.set_params()
    if not np.all(depth_gt.shape == depth_res.shape):
        depth_gt = cv2.resize(depth_gt, (depth_res.shape[1],
            depth_res.shape[0]), interpolation=cv2.INTER_NEAREST)

    sample_rate = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
    acc = np.zeros(len(sample_rate), dtype=np.float32)
    uts.plot_images({'image': depth_gt})

    acc_o = eval_depth([depth_res], [depth_gt])

    for i, rate in enumerate(sample_rate):
        depth_gt_down = uts_3d.down_sample_depth(depth_gt,
                                                 method='uniform',
                                                 percent=rate,
                                                 K=params['intrinsic'])
        depth = uts_3d.xyz2depth(depth_gt_down,
                                 params['intrinsic'],
                                 depth_gt.shape)
        depth_up = upsampler.LaplacianDeform(depth_res, depth_gt_down,
                params['intrinsic'], False)

        acc[i] = eval_depth([depth_up], [depth_gt])

    if vis:
        plot_figure(np.append(0, sample_rate),
                    np.apend(acc_o, acc),
                    'depth_acc', 'sample rate',
                    'relative l1 error')
    else:
        print "rates: {}, thresholds {}".format(sample_rate, acc)


def main(argv):
    argv = FLAGS(argv)
    test_single()


if __name__ == '__main__':
    # pyplot.plot([0.01, 0.05, 0.1, 0.2, 0.4, 0.8],
    #             [0.05312619,  0.04010963,
    #              0.03726558,  0.0369869,   0.03562214,  0.0342297],
    #             color=cm.jet(1.0),
    #             label='depth_acc')
    # pyplot.xlabel('sample rate')
    # pyplot.legend(loc='relative l1 error')
    # pyplot.show()

    main(sys.argv)
