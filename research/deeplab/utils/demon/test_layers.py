import sys
paddle_root = '/home/peng/Paddle/'
sys.path.insert(0, paddle_root + 'python')
sys.path.insert(0, paddle_root + 'paddle/')
sys.path.insert(1, "./")

import numpy as np
import paddle.v2 as pd
import cv2
import layers.cost_layers as cost_layers

pd.init(use_gpu=True, gpu_id=0)
def test_inverse(argv):
    depth_np = np.array([-3, 2, 2, 2,
                         1, 2, 4, 4,
                         4, 2, 2, 4,
                         4, 4, 2, 2], dtype=np.float32)

    # depth_np = 4 * np.ones((4, 4), dtype=np.float32)
    depth_np = depth_np.flatten()

    height=4
    width=4
    depth = pd.layer.data(
            name="depth", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)

    inv_depth = pd.layer.mixed(input=[pd.layer.identity_projection(
                                      input=depth)],
                               act=pd.activation.Inv())

    parameters, topo = pd.parameters.create(inv_depth)
    inv_depth_np = pd.infer(
            output=inv_depth,
            parameters=parameters,
            input=[(depth_np, )],
            feeding={'depth':0})

    print inv_depth_np



def test_one_hot(argv):
    label_np = np.array([1, 2, 2, 0,
                         1, 2, 4, 4,
                         4, 2, 2, 0,
                         4, 4, 2, 2], dtype=np.float32)

    height=4
    width=4
    label = pd.layer.data(
            name="label",
            type=pd.data_type.dense_vector(height*width),
            height=height, width=width)

    label = pd.layer.resize(input=label, size=1)
    one_hot = pd.layer.one_hot(input=label, class_num=5)
    parameters, topo = pd.parameters.create(one_hot)
    one_hot_np = pd.infer(
            output=topo,
            parameters=parameters,
            input=[(label_np, )],
            feeding={'label':0})

    print one_hot_np


def test_gradient_weight(argv):
    weight_np = np.array([1, 1, 1, 0,
                         1, 1, 1, 1,
                         1, 1, 1, 0,
                         1, 1, 1, 1], dtype=np.float32)
    scales = [1]
    height = 4
    width = 4

    weight = pd.layer.data(
            name="weight", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)

    weight_diff = pd.layer.gradient_diff(input=weight, scales=scales)
    weight_diff = util_layers.reduce(input=weight_diff,
                                     shape=[len(scales) * 2, height, width],
                                     op='sum')
    weight_diff = util_layers.math_op(input=weight_diff,
                                      act=pd.activation.IsZero())
    weight_diff = util_layers.math_op(input=[weight_diff, weight], op='dot')
    parameters, topo = pd.parameters.create(weight_diff)

    weight_diff_np = pd.infer(
            output=topo,
            parameters=parameters,
            input=[(weight_np, )],
            feeding={'weight':0})

    print weight_diff_np


def test_is_zero(argv):

    depth_np = np.array([1, 2, 2, 0,
                         1, 2, 4, 4,
                         4, 2, 2, 0,
                         4, 4, 2, 2], dtype=np.float32)

    height=4
    width=4

    depth = pd.layer.data(
            name="depth", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)

    mask = pd.layer.mixed(input=[pd.layer.identity_projection(
                                      input=depth)],
                               act=pd.activation.IsZero())

    parameters, topo = pd.parameters.create(mask)

    mask_np = pd.infer(
            output=topo,
            parameters=parameters,
            input=[(depth_np, )],
            feeding={'depth':0})

    print mask_np


def test_gradient_dff(argv):
    depth_np = np.array([1, 2, 2, 2,
                         1, 2, 4, 4,
                         4, 2, 2, 4,
                         4, 4, 2, 2], dtype=np.float32)

    height=48
    width=64

    depth_np = depth_np.reshape((4, 4))
    depth_np = cv2.resize(depth_np, (width, height))
    depth_np = depth_np.flatten()

    depth = pd.layer.data(
            name="depth", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)

    gradient_diff = pd.layer.gradient_diff(input=depth,
                                           scales=[1, 2])

    parameters, topo = pd.parameters.create(gradient_diff)
    gradient_diff_np = pd.infer(
            output=topo,
            parameters=parameters,
            input=[(depth_np, ), (depth_np, )],
            feeding={'depth':0})

    print gradient_diff_np


def test_trans_depth_flow(argv):

    depth_np = np.array([1, 2, 2, 2,
                         1, 2, 4, 4,
                         4, 2, 2, 4,
                         4, 4, 2, 2], dtype=np.float32)
    trans_np= np.array([1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0], dtype=np.float32)

    height=4
    width=4
    depth = pd.layer.data(
            name="depth", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)

    trans = pd.layer.data(
            name="trans", type=pd.data_type.dense_vector(10))

    inv_depth = pd.layer.mixed(input=[pd.layer.identity_projection(
                                      input=depth)],
                               act=pd.activation.Inv())

    flow_trans = pd.layer.trans_depth_flow(input=[depth, trans],
                                           depth2flow=True)
    depth_trans = pd.layer.trans_depth_flow(input=[flow_trans, trans],
                                            depth2flow=False)

    parameters, topo = pd.parameters.create(depth_trans)

    flow2, depth2 = pd.infer(
            output=[flow_trans, depth_trans],
            parameters=parameters,
            input=[(depth_np, trans_np)],
            feeding={'depth':0, 'trans':1})

    print flow2, depth2


def test_ns_l2_cost(argv):
    flow_np = np.array([1, 2, 2, 2,
                        1, 2, 4, 4,
                        4, 2, 2, 4,
                        4, 4, 2, 2], dtype=np.float32)
    flow_gt_np = np.array([1, 2, 2, 2,
                           1, 2, 4, 4,
                           4, 2, 2, 4,
                           4, 4, 2, 2], dtype=np.float32)
    flow_gt_np = flow_gt_np + 1.0

    height=2
    width=4
    channel=2

    weight_np = np.ones((height, width), dtype=np.float32).flatten()
    flow = pd.layer.data(
            name="flow", type=pd.data_type.dense_vector(2*height*width),
            height=height, width=width)
    flow_gt = pd.layer.data(
            name="flow_gt", type=pd.data_type.dense_vector(2*height*width),
            height=height, width=width)
    weight = pd.layer.data(
            name="weight", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)

    # cost = cost_layers.math_op(input=flow, act=pd.activation.Sqrt())
    cost = cost_layers.ns_ele_l2_cost(input=flow, label=flow_gt, weight=weight,
                                      height=height, width=width, num_channel=channel)

    parameters, topo = pd.parameters.create(cost)
    cost = pd.infer(
            output=cost,
            parameters=parameters,
            input=[(flow_np, flow_gt_np, weight_np)],
            feeding={'flow':0, 'flow_gt':1, 'weight':2})

    print cost.shape, cost


def test_ele_norm_cost(argv):
    flow_np = np.array([1, 2, 2, 2,
                        1, 2, 4, 4,
                        4, 2, 2, 4,
                        4, 4, 2, 2], dtype=np.float32)
    flow_gt_np = flow_np + 2

    height=4
    width=4
    channel=1

    weight_np = np.ones((height, width), dtype=np.float32).flatten()
    flow = pd.layer.data(
            name="flow", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)
    flow_gt = pd.layer.data(
            name="flow_gt", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)
    weight = pd.layer.data(
            name="weight", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)

    # cost = cost_layers.math_op(input=flow, act=pd.activation.Sqrt())
    cost = cost_layers.ele_norm_cost(input=flow, label=flow_gt, weight=weight,
                                     height=height, width=width, num_channel=channel,
                                     cost_type='l2')

    parameters, topo = pd.parameters.create(cost)
    cost_np = pd.infer(
            output=topo,
            parameters=parameters,
            input=[(flow_np, flow_gt_np, weight_np)],
            feeding={'flow':0, 'flow_gt':1, 'weight':2})

    print cost_np


def test_relative_l1(argv):
    depth_np = np.array([1, 2, 2, 2,
                        1, 2, 4, 4,
                        4, 2, 2, 4,
                        4, 4, 2, 2], dtype=np.float32)

    depth_gt_np = np.array([1, 2, 2, 2,
                           1, 2, 4, 4,
                           4, 2, 2, 4,
                           4, 4, 2, 2], dtype=np.float32)
    depth_gt_np = depth_gt_np + 1.0

    height=4
    width=4
    channel=1

    weight_np = np.ones((height, width), dtype=np.float32).flatten()
    depth = pd.layer.data(
            name="depth", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)
    depth_gt = pd.layer.data(
            name="depth_gt", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)
    weight = pd.layer.data(
            name="weight", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)

    # cost = cost_layers.math_op(input=flow, act=pd.activation.Sqrt())
    cost = cost_layers.relative_l1(input=depth, label=depth_gt, weight=weight,
                                   height=height, width=width, num_channel=channel)

    parameters, topo = pd.parameters.create(cost)
    cost = pd.infer(
            output=cost,
            parameters=parameters,
            input=[(depth_np, depth_gt_np, weight_np)],
            feeding={'depth':0, 'depth_gt':1, 'weight':2})

    print cost.shape, cost


def test_angle_error(argv):

    normal_np = np.array([1, 2, 2, 2,
                          1, 2, 4, 4,
                          4, 2, 2, 4,
                          4, 4, 2, 2,
                          4, 2, 2, 4,
                          4, 4, 2, 2], dtype=np.float32)

    normal_gt_np = np.array([1, 2, 2, 2,
                             1, 2, 4, 4,
                             4, 2, 2, 4,
                             4, 4, 2, 2,
                             4, 2, 2, 4,
                             4, 4, 2, 2], dtype=np.float32)

    normal_gt_np = normal_gt_np + 1.0
    height=2
    width=4
    channel=3

    weight_np = np.ones((height, width), dtype=np.float32).flatten()
    normal = pd.layer.data(
            name="normal", type=pd.data_type.dense_vector(3*height*width),
            height=height, width=width)
    normal_gt = pd.layer.data(
            name="normal_gt", type=pd.data_type.dense_vector(3*height*width),
            height=height, width=width)
    weight = pd.layer.data(
            name="weight", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)

    cost = cost_layers.inner_product_cost(input=normal,
                                          label=normal_gt,
                                          weight=weight,
                                          height=height,
                                          width=width,
                                          num_channel=channel,
                                          is_angle=True)

    parameters, topo = pd.parameters.create(cost)
    cost_np = pd.infer(
            output=cost,
            parameters=parameters,
            input=[(normal_np, normal_gt_np, weight_np)],
            feeding={'normal':0, 'normal_gt':1, 'weight':2})

    print cost_np


def test_classification_error(argv):
    label_np = np.array([1, 2, 2, 0,
                         1, 2, 4, 4,
                         4, 2, 2, 0,
                         4, 4, 2, 2], dtype=np.float32)

    output_np = np.array([0, 2, 2, 0,
                         0, 2, 4, 4,
                         3, 2, 2, 0,
                         3, 4, 2, 2], dtype=np.float32)

    height=4
    width=4
    class_num=5

    weight_np = np.ones((height, width), dtype=np.float32).flatten()
    output = pd.layer.data(
            name="output", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)
    label = pd.layer.data(
            name="label", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)
    weight = pd.layer.data(
            name="weight", type=pd.data_type.dense_vector(height*width),
            height=height, width=width)

    # cost = cost_layers.math_op(input=flow, act=pd.activation.Sqrt())
    acc = cost_layers.pixel_accuracy(
      input=output, label=label, weight=weight,
      height=height, width=width)

    iou = cost_layers.iou_score(
      input=output, label=label, weight=weight,
      height=height, width=width,
      class_num=class_num, is_cost=False)

    parameters, topo = pd.parameters.create([acc, iou])
    acc_np, iou_np = pd.infer(
            output=topo,
            parameters=parameters,
            input=[(output_np, label_np, weight_np)],
            feeding={'output':0, 'label':1, 'weight':2})

    print acc_np, iou_np


if __name__ == '__main__':
    # test_ns_l2_cost(sys.argv)
    # test_one_hot(sys.argv)
    test_classification_error(sys.argv)
    # test_ele_norm_cost(sys.argv)
    # test_relative_l1(sys.argv)
    # test_angle_error(sys.argv)
    # test_gradient_dff(sys.argv)
    # test_is_zero(sys.argv)
    # test_gradient_weight(sys.argv)

