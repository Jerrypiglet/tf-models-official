import matplotlib.pyplot as plt
import utils.utils as uts
import argparse
import logging
import time

def plot_log(logfile, item_names):
    for item_name in item_names:
        uts.plot_mxnet_log(logfile, item_name)

def plot_log_online(logfile, item_name='loss'):
    plt.ion()
    fig = plt.figure(figsize=(10, 5))

    while True:
        uts.plot_mxnet_log(logfile, item_name,
                           fig=fig)
        time.sleep(20)
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a mx model.')
    parser.add_argument('--logfile', default=None,
        help='the log file want to watch')
    parser.add_argument('--is_online', type=uts.str2bool, default=True,
        help='true means continue training.')
    parser.add_argument('--item_names', default='loss',
        help='true means continue training.')

    args = parser.parse_args()
    logging.info(args)
    if args.is_online:
        plot_log_online(args.logfile)
    else:
        names = args.item_names.split(',')
        plot_log(args.logfile, names)

