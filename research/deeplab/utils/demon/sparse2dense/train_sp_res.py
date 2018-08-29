#
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
import layers.util_layers as util_layers
import network.demon_net as d_net

FLAGS = gflags.FLAGS
gflags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for model')
gflags.DEFINE_integer('gpu_id', 0, 'Gpu id used in the training')
gflags.DEFINE_integer('trainer_count', 1, 'number of gpu use in training')
gflags.DEFINE_integer('buffer_size', 100, 'buffer size for training')
gflags.DEFINE_string('init_model', 'output/tf_model_full_5.tar.gz', 'init model name')
gflags.DEFINE_string('learning_loss', 'nqmse', 'Learning type of loss for model')


def gen_cost(outputs, gts, loss, params):
    cost = []
    stage = params['stage']

    if 'depth' in loss:
        cost_depth = cost_layers.ele_norm_cost(input=outputs['depth_inv'],
                                         label=gts['depth_gt'],
                                         weight=gts['weight'],
                                         height=params['size_stage'][1][0],
                                         width=params['size_stage'][1][1],
                                         num_channel=1,
                                         cost_type='l1')
        cost.append(cost_depth)


    if 'normal' in loss:
        height, width = params['size_stage'][1]
        normal = util_layers.norm(outputs['normal'], height, width, 3)
        label = paddle.layer.bilinear_interp(input=gts['normal_gt'],
            out_size_x=width, out_size_y=height)
        label = util_layers.norm(label, height, width, 3)
        cost_normal = cost_layers.ns_ele_l2_cost(input=normal,
                                         label=label,
                                         weight=gts['weight'],
                                         height=params['size_stage'][1][0],
                                         width=params['size_stage'][1][1],
                                         num_channel=3)
        # cost_normal = cost_layers.inner_product_cost(input=outputs['normal'],
        #                                  label=gts['normal'],
        #                                  weight=gts['weight'],
        #                                  height=params['size_stage'][1][0],
        #                                  width=params['size_stage'][1][1],
        #                                  num_channel=3)
        cost.append(cost_normal)


    if 'trans' in loss:
        cost_rotation = cost_layers.inner_product_cost(
            input=outputs['rotation'],
            label=gts['rotation'],
            weight=None,
            height=1, width=1, num_channel=3)
        cost_translation = cost_layers.ele_norm_cost(
            input=outputs['translation'],
            label=gts['translation'],
            weight=None,
            height=1, width=1, num_channel=3, cost_type='l1')
        cost.append(cost_rotation)
        cost.append(cost_translation)

    return  cost


def get_feeding(tasks, start_id=4):
    feeding = []
    if 'flow' in tasks:
        feeding = ['flow']
    if 'trans' in tasks:
        feeding = feeding + ['rotation', 'translation']
    if 'depth' in tasks:
        feeding = feeding + ['depth']
    if 'normal' in tasks:
        feeding = feeding + ['normal']

    gt_name = [x + '_gt' for x in feeding]
    return dict(zip(gt_name, range(4, len(gt_name) + 4)))


def train():
    # PaddlePaddle init, gpu_id=FLAGS.gpu_id
    paddle.init(use_gpu=True, trainer_count=4, gpu_id=FLAGS.gpu_id)
    # paddle.init(use_gpu=True, trainer_count=2)
    data_source = 'sun3d'
    tasks = ['flow', 'trans', 'depth', 'normal']
    tasks = ['flow', 'depth']
    tasks = ['normal']
    feeding_task = get_feeding(tasks)

    params = sun3d.set_params()
    params['stage'] = 2
    inputs = d_net.get_demon_inputs(params)
    gts = d_net.get_ground_truth(params)

    # Add neural network config
    outputs, out_field = d_net.get_demon_outputs(inputs, params)
    cost = gen_cost(outputs, gts, tasks, params)

    # Create parameters
    print "Loading pre trained model"
    parameters = paddle.parameters.create(cost)

    if FLAGS.init_model:
        with gzip.open(FLAGS.init_model, 'r') as f:
            parameters_init = paddle.parameters.Parameters.from_tar(f)
        for name in parameters.names():
            parameters.set(name, parameters_init.get(name))

    # # Create optimizer poly learning rate
    # momentum_optimizer = paddle.optimizer.Momentum(
    #     momentum=0.9,
    #     regularization=paddle.optimizer.L2Regularization(rate=0.0002 * params['batch_size']),
    #     learning_rate=0.1 / params['batch_size'],
    #     learning_rate_decay_a=0.1,
    #     learning_rate_decay_b=50000 * 100,
    #     learning_rate_schedule='discexp',
    #     batch_size=params['batch_size'])

    # Create optimizer poly learning rate
    adam_optimizer = paddle.optimizer.Adam(
        beta1=0.9,
        learning_rate=0.000015 / params['batch_size'],
        learning_rate_decay_a=0.8,
        learning_rate_decay_b=100000,
        learning_rate_schedule='discexp',
        regularization=paddle.optimizer.L2Regularization(
            rate=0.0002 * params['batch_size']),
        batch_size=params['batch_size'] * FLAGS.trainer_count)

    # End batch and end pass event handler
    feeding = {'image1': 0, 'image2': 1, 'weight': 2, 'intrinsic': 3}
    feeding.update(feeding_task)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 50 == 0:
                print "\nPass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        elif isinstance(event, paddle.event.EndPass):
            result = trainer.test(
                reader=paddle.batch(sun3d.test(params['test_scene'][0:4],
                                               height=params['size'][0],
                                               width=params['size'][1],
                                               tasks=tasks),
                                    batch_size=2*params['batch_size']),
                feeding=feeding)

            task_string = '_'.join(tasks)
            print "\nTask %s, Pass %d, Cost %f" % (task_string,
                    event.pass_id, result.cost)

            folder = params['output_path'] + '/' + data_source
            uts.mkdir_if_need(folder)
            model_name = folder + '/model_stage_' + str(params['stage']) + '_' + task_string + '.tar.gz'

            with gzip.open(model_name, 'w') as f:
                parameters.to_tar(f)
            print "\nsave with pass %d" % (event.pass_id)

    # Create trainer
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=adam_optimizer)

    reader = sun3d.train(scene_names=params['train_scene'],
                         height=params['size'][0],
                         width=params['size'][1],
                         tasks=tasks)

    batch_reader = paddle.batch(paddle.reader.shuffle(reader,
                                buf_size=FLAGS.buffer_size),
                                batch_size=params['batch_size'])

    trainer.train(
            reader=batch_reader,
            num_passes=100,
            event_handler=event_handler,
            feeding=feeding)


def main(argv):
    argv=FLAGS(argv) # parse argv to FLAG
    train()

if __name__ == '__main__':
    main(sys.argv)
