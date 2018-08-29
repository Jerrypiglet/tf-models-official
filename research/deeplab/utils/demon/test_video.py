import gflags
import sys

paddle_root = '/home/peng/Paddle/'
sys.path.insert(0, paddle_root + 'python')
sys.path.insert(0, paddle_root + 'paddle/')
sys.path.insert(1, "./")

import pdb
import os
import cv2
import gzip
import numpy as np
import paddle.v2 as paddle
import paddle.trainer.config_parser as cp
import time

import data.sun3d as sun3d
import utils.utils as uts
from utils.vis import visualize_prediction

import layers.cost_layers as cost_layers
import network.demon_net as d_net
from paddle.utils import preprocess_util
from collections import OrderedDict
import DemonPredictor as dp
import matplotlib.pyplot as plt

gflags.DEFINE_integer('gpu_id', 1, 'Gpu id used')
FLAGS = gflags.FLAGS

def test_video():
    # PaddlePaddle init
    cv2.namedWindow("frame")
    cv2.namedWindow("depth")
    cv2.namedWindow("normal")
    base_path = '/home/peng/Data/videos/'
    video_names = preprocess_util.list_files(base_path)
    prefix_len = len(base_path)
    for name in video_names[0:]:
        name = name[prefix_len:]
        output_path = base_path + name[:-4] + '/'
        if os.path.exists(output_path):
            continue
        video_path = base_path + name
        uts.save_video_to_images(video_path, output_path, max_frame=20000)

    return

    paddle.init(use_gpu=True, gpu_id=FLAGS.gpu_id)
    params = sun3d.set_params()
    params['demon_model'] = 'output/tf_model_full_5.tar.gz'
    inputs = d_net.get_demon_inputs(params)
    geo_predictor = dp.DeMoNGeoPredictor(params, is_init=True)

    # load in video
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    frame_step = 10
    ret, last_frame = cap.read()
    height, width = last_frame.shape[0:2]

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_id += 1
        if frame_id % frame_step == 0:
            # pdb.set_trace()
            flow, normal, depth, motion = geo_predictor.demon_geometry(
                frame, last_frame)
            last_frame = frame

            depth = cv2.resize(depth, (width / 2, height / 2))
            depth = depth / np.amax(depth)

            normal = cv2.resize(normal, (width / 2, height / 2))
            normal = (normal + 1.) / 2.
            two_frame = np.concatenate([frame, last_frame], axis=1)
            cv2.imshow('frame', two_frame)
            cv2.imshow('depth', depth)
            cv2.imshow('normal', normal)
            print motion

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # register different result through
    # fusion and recover true motion of the camers


def vis_video():
    # cv2.namedWindow("frame")
    # cv2.namedWindow("depth")
    # cv2.namedWindow("normal")
    base_path = '/home/peng/Data/videos/'
    video_names = preprocess_util.list_files(base_path)
    prefix_len = len(base_path)
    rotate_scene = ['chairs', 'cubic1', 'cubic2', 'hall1', 'hall2', 'printer']
    plt.ion()

    for name in video_names:
        name = name[prefix_len:-4]
        print name
        video_path = base_path + name + '/'
        res_path = base_path + name + '_res/'
        max_frame = len(os.listdir(video_path))
        frame_id = 10
        frame_step = 10

        while(frame_id + frame_step < max_frame):

            frame = cv2.imread(
                video_path + 'frame {}.jpg'.format(frame_id))
            if name in rotate_scene:
                frame = np.rot90(frame, 3)

            height, width = frame.shape[0:2]
            frame = cv2.resize(frame, (int(width/2), int(height/2)))

            if not os.path.exists(res_path + 'depth_frame_{}.npy'.format(frame_id)):
                break

            depth = np.load(res_path + 'depth_frame_{}.npy'.format(frame_id))
            normal= np.load(res_path + 'normal_frame_{}.npy'.format(frame_id))
            normal = (normal + 1.0) / 2.0
            normal[normal < 0.] = 0.
            depth = depth / np.amax(depth)

            cv2.imshow('frame', frame)
            cv2.imshow('depth', depth)
            cv2.imshow('normal', normal)
            # uts.plot_images(OrderedDict([('frame',frame),
            #                              ('depth',depth),
            #                              ('normal',normal)]), layout=(1, 3))
            plt.pause(0.5)
            # plt.clf()
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            if frame_id == 10:
                raw_input('Press to continue ...')
            frame_id += frame_step


def main(argv):
    argv = FLAGS(argv)
    vis_video()
    # test_video()

if __name__ == '__main__':
    main(sys.argv)
