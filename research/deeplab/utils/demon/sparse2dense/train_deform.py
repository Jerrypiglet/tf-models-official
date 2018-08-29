import gflags
import sys
paddle_root = '/home/peng/Paddle/'
sys.path.insert(0, paddle_root + 'python')
sys.path.insert(0, paddle_root + 'paddle/')
sys.path.insert(1, "../")

import paddle.v2 as paddle
import numpy as np
import data.sun3d as sun3d
import gzip
import utils.utils as uts
import layers.cost_layers as cost_layers
import layers.util_layers as util_layers
import network.demon_net as d_net
import network.upsample_net as u_net


FLAGS = gflags.FLAGS
gflags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for model')
gflags.DEFINE_integer('buffer_size', 100, 'buffer size for training')
gflags.DEFINE_integer('num_passes', 200, 'buffer size for training')

gflags.DEFINE_string('gpu_id', '0', 'Gpu id used in the training')
gflags.DEFINE_string('suffix', '', 'suffix for model id')
gflags.DEFINE_string('init_model', 'output/tf_model_full_5.tar.gz', 'init model name')
gflags.DEFINE_string('learning_loss', 'nqmse', 'Learning type of loss for model')
gflags.DEFINE_boolean('is_test', False, 'whether to test results')


def gen_cost(outputs, gts, params,
             cost_name=['depth_l1', 'depth_gradient'],
             weights=[1.0, 0.4],
             is_inverse=False):

    suffix=''
    if is_inverse:
        suffix='_inv'
    in_depth_name = 'depth' + suffix
    gt_depth_name = 'depth_gt' + suffix

    # depth loss
    cost = []
    if 'depth_l1' in cost_name:
        cost_depth = cost_layers.ele_norm_cost(input=outputs[in_depth_name],
                                               label=gts[gt_depth_name],
                                               weight=gts['weight'],
                                               height=params['size'][0],
                                               width=params['size'][1],
                                               num_channel=1,
                                               cost_type='l1')
        cost_depth = util_layers.mul(cost_depth, weights[0])
        cost.append(cost_depth)


    if 'rel_l1' in cost_name:
        cost_depth = cost_layers.relative_l1(input=outputs[in_depth_name],
                                             label=gts[gt_depth_name],
                                             weight=gts['weight'],
                                             height=params['size'][0],
                                             width=params['size'][1])
        cost_depth = util_layers.mul(cost_depth, weights[0])
        cost.append(cost_depth)


    if 'depth_gradient' in cost_name:
        cost_depth_gradient = cost_layers.gradient_cost(
                                             input=outputs[in_depth_name],
                                             label=gts[gt_depth_name],
                                             weight=gts['weight'],
                                             height=params['size'][0],
                                             width=params['size'][1],
                                             num_channel=1,
                                             scales=[1,2,4,8])
        cost_depth_gradient = util_layers.mul(cost_depth_gradient, weights[1])
        cost.append(cost_depth_gradient)

    return  cost


def train(is_test=True):
    # PaddlePaddle init, gpu_id=FLAGS.gpu_id
    gpu_ids = [int(x) for x in FLAGS.gpu_id.split(',')]
    trainer_count = len(gpu_ids)
    # cost_name = ['depth_l1', 'depth_gradient']
    cost_name = ['rel_l1', 'depth_gradient']
    # cost_name = ['rel_l1']
    is_inverse = False

    if len(gpu_ids) == 1:
        gpu_ids = gpu_ids[0]
        paddle.init(use_gpu=True, gpu_id=gpu_ids)
    else:
        paddle.init(use_gpu=True, trainer_count=trainer_count)

    # paddle.init(use_gpu=True, trainer_count=2)
    data_source = 'sun3d'
    tasks = ['depth']

    if data_source == 'sun3d':
        params = sun3d.set_params()

    inputs = u_net.get_inputs(params)
    gts = u_net.get_ground_truth(params)
    outputs = u_net.refine_net(inputs, params)
    cost = gen_cost(outputs, gts, params, cost_name)

    # Create parameters
    parameters, _ = paddle.parameters.create(cost)

    # Create optimizer poly learning rate
    optimizer = paddle.optimizer.Adam(
        beta1=0.9,
        learning_rate=FLAGS.learning_rate / params['batch_size'],
        learning_rate_decay_a=0.8,
        learning_rate_decay_b=10000,
        learning_rate_schedule='discexp',
        regularization=paddle.optimizer.L2Regularization(
            rate=0.0002 * params['batch_size']),
        batch_size=params['batch_size'] * trainer_count)

    # optimizer = paddle.optimizer.Momentum(
    #     learning_rate=FLAGS.learning_rate / params['batch_size'],
    #     momentum=0.9,
    #     learning_rate_decay_b=50000,
    #     learning_rate_schedule='discexp',
    #     regularization=paddle.optimizer.L2Regularization(
    #         rate=0.0005 * params['batch_size']))

    # End batch and end pass event handler
    if is_inverse:
        feeding = {'image1': 0,
                   'depth_inv': 1,
                   'depth_gt_inv': 2,
                   'weight': 3}
    else:
        feeding = {'image1': 0,
                   'depth': 1,
                   'depth_gt': 2,
                   'weight': 3}

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 50 == 0:
                if not isinstance(event.cost, list):
                    cost = [event.cost]
                else:
                    cost = event.cost
                print "\nPass %d, Batch %d, " % (event.pass_id, event.batch_id),
                for i in range(len(cost)):
                    print "%s: %f, " %  (cost_name[i], cost[i]),
                print "\n"
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        elif (isinstance(event, paddle.event.EndPass) and \
              event.pass_id % 4 == 1):
            print "Testing",
            result = trainer.test(
                reader=paddle.batch(sun3d.test_upsampler(
                    params['test_scene'][0:5],
                    rate=params['sample_rate'],
                    height=params['size'][0],
                    width=params['size'][1]),
                    batch_size=params['batch_size']),
                feeding=feeding)

            print "\nTask upsample, Pass %d," % (event.pass_id),
            if not isinstance(result.cost, list):
                cost = [result.cost]
            else:
                cost = result.cost
            for i in range(len(cost)):
                print "%s: %f, " %  (cost_name[i], cost[i]),

            folder = params['output_path'] + '/upsampler/'
            uts.mkdir_if_need(folder)
            model_name = folder + '/upsample_model_' + \
                         FLAGS.suffix + '.tar.gz'
            with gzip.open(model_name, 'w') as f:
                parameters.to_tar(f)
            print "model saved at %s" % model_name

    # Create trainer
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

    if is_test:
        print("load parameters from {}".format(FLAGS.init_model))
        with gzip.open(FLAGS.init_model, 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)

        result = trainer.test(
            reader=paddle.batch(sun3d.test_upsampler(
                params['test_scene'][0:5],
                rate=params['sample_rate'],
                height=params['size'][0],
                width=params['size'][1]),
                batch_size=params['batch_size']),
                feeding=feeding)
        print "Test cost {}\n".format(result.cost)

    else:
        reader = sun3d.train_upsampler(scene_names=params['train_scene'],
                                       rate=params['sample_rate'],
                                       height=params['size'][0],
                                       width=params['size'][1],
                                       max_num=32)

        batch_reader = paddle.batch(paddle.reader.shuffle(reader,
                                    buf_size=FLAGS.buffer_size),
                                    batch_size=params['batch_size'])
        trainer.train(reader=batch_reader,
                      num_passes=FLAGS.num_passes,
                      event_handler=event_handler,
                      feeding=feeding)


def main(argv):
    argv=FLAGS(argv) # parse argv to FLAG
    train(FLAGS.is_test)

if __name__ == '__main__':
    main(sys.argv)
