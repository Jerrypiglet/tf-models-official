import numpy as np
import pdb
import mxnet as mx
import Networks.net_util as net_util
import logging

def surgury_net(model_name, out_name, name1='pose', name2='critic'):
    arg_params, _ = net_util.load_mxparams_from_file(model_name + '.params')

    for name, arr in arg_params.items():
        if name1 in name:
            name_new = name
            name_new = name_new.replace(name1, name2)
            arg_params[name_new] = arr
            del arg_params[name]

    print arg_params.keys()
    mx.ndarray.save(out_name + '.params', arg_params)

def surgury_net_2(model_name, out_name):
    arg_params, _ = net_util.load_mxparams_from_file(model_name + '.params')
    name = 'pose_fc_1_weight'
    temp = arg_params[name].copy()
    arg_params[name] = mx.nd.zeros((1024, 26624))
    arg_params[name][:, :15360] = temp
    mx.ndarray.save(out_name + '.params', arg_params)


def surgury_resnext(model_name, out_name):
    arg_params, aux_params = net_util.load_mxparams_from_file(model_name + '.params')
    for name, val in arg_params.items():
        print name, val.shape

    pdb.set_trace()
    name = 'conv0_weight'
    temp = arg_params[name].copy()
    arg_params[name] = mx.nd.tile(temp, (1, 4, 1, 1))
    mx.ndarray.save(out_name + '.params', arg_params)



if __name__ == '__main__':
    network = './output/resnext-50-0000'
    out_net = './output/resnext-50-pose-0000'
    surgury_resnext(network, out_net)
    # network = './output/policy-apollo-rel-0006'
    # name1 = 'pose'
    # # network_save = './output/policy-demon-
    # network_save = './output/policy-pg-init'
    # name2 = 'actor'
    # logging.info(network_save)
    # surgury_net(network, network_save, name1, name2)
    # surgury_net_2(network, network_save)


    # network_save = './output/value-net-init'
    # name2 = 'critic'
    # surgury_net(network, network_save, name1, name2)

