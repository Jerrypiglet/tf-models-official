""" Test over different dataset test set for evaluate different task results """
import gflags

import sys
paddle_root = '/home/peng/Paddle/'
sys.path.insert(0, paddle_root + 'python')
sys.path.insert(0, paddle_root + 'paddle/')
sys.path.insert(1, "./")

import paddle.v2 as paddle
import numpy as np
import data.sun3d as sun3d
import gzip
import utils.utils as uts
import layers.cost_layers as cost_layers
import network.demon_net as d_net

FLAGS = gflags.FLAGS
gflags.DEFINE_integer('gpu_id', 0, 'Gpu id used in the training')
gflags.DEFINE_string('init_model', './output/tf_model_full_5.tar.gz', 'init model name')


def gen_cost(outputs, gts, loss, params):
    #the cost of flow is equivalent with the evaluation EPE
    cost = []
    if 'flow' in loss:
        cost_flow = cost_layers.ns_ele_l2_cost(input=outputs['flow'],
                                               label=gts['flow_gt'],
                                               weight=gts['weight'],
                                               height=params['size'][0],
                                               width=params['size'][1],
                                               num_channel=2)
        cost.append(cost_flow)

    if 'depth' in loss:
        # we set the
        depth_gt = pd.layer.mixed(input=[pd.layer.identity_projection(
                                         input=gts['depth_gt'])],
                                  act=pd.activation.Inv())
        cost_depth = cost_layers.relative_l1(input=outputs['depth'],
                                             label=depth_gt,
                                             weight=gts['weight'],
                                             height=params['size'][0],
                                             width=params['size'][1])
        cost.append(cost_depth)

    if 'normal' in loss:
        cost_normal = cost_layers.inner_product_cost(input=outputs['normal'],
                                                     label=gts['normal_gt'],
                                                     weight=gts['weight'],
                                                     height=params['size'][0],
                                                     width=params['size'][1],
                                                     num_channel=3,
                                                     is_angle=True)
        cost.append(cost_normal)

        return  cost


def test():
    # PaddlePaddle init, gpu_id=FLAGS.gpu_id
    paddle.init(use_gpu=True, gpu_id=FLAGS.gpu_id)
    # eval_tasks = {'flow':0, 'depth':1, 'normal':2}
    tasks = ['flow', 'depth']
    tasks = ['flow', 'depth', 'normal']
    # tasks = ['normal']
    gt_name = [x + '_gt' for x in tasks]
    dict_task = dict(zip(gt_name, range(4, 4 + len(tasks))))

    params = sun3d.set_params()
    params['stage'] = 2
    # params['stage'] = 4

    inputs = d_net.get_demon_inputs(params)
    gts = d_net.get_ground_truth(params)

    #Add neural network config
    outputs, out_field = d_net.get_demon_outputs(inputs, params, ext_inputs=None)
    cost = gen_cost(outputs, gts, tasks, params)
    parameters = paddle.parameters.create(layers=cost)

    print("load parameters from {}".format(FLAGS.init_model))
    # if FLAGS.init_model:
    #     with gzip.open(FLAGS.init_model, 'r') as f:
    #         parameters_init = paddle.parameters.Parameters.from_tar(f)
    #     for name in parameters.names():
    #         parameters.set(name, parameters_init.get(name))

    optimizer = paddle.optimizer.Momentum(
        learning_rate=0,
        momentum=0,
        regularization=paddle.optimizer.L2Regularization(rate=0.0))
    trainer = paddle.trainer.SGD(cost=cost,
                parameters=parameters, update_equation=optimizer)

    feeding = {'image1': 0, 'image2': 1, 'weight': 2, 'intrinsic': 3}
    feeding.update(dict_task)

    print("start inference and evaluate")
    result = trainer.test(
        reader=paddle.batch(sun3d.test(params['test_scene'][0:2],
                                       height=params['size'][0],
                                       width=params['size'][1],
                                       tasks=tasks),
                            batch_size=32),
        feeding=feeding)
    print "Test with task {} and cost {}\n".format(tasks, result.cost)


def main(argv):
    argv=FLAGS(argv) # parse argv to FLAG
    test()

if __name__ == '__main__':
    main(sys.argv)
