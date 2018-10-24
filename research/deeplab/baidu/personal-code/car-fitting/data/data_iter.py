import pdb
import mxnet as mx
import numpy as np
from imgaug import augmenters as iaa
from mxnet.io import DataIter, DataBatch

from collections import OrderedDict, namedtuple
Batch = namedtuple('Batch', ['data'])


class GNNIter(DataIter):
    """
    Parameters
    ----------

    """
    def __init__(self,
                 params,
                 config,
                 set_name='train',
                 data_type='mx',
                 data_setting=None,
                 label_setting=None):

        super(GNNIter, self).__init__()
        self.batch_size = config.batch_size
        self.array = mx.nd.empty if data_type == 'mx' \
                else np.empty

        self.data_name = data_setting.keys()
        self.label_name = label_setting.keys()
        self.data_setting = data_setting
        self.label_setting = label_setting

        list_name = '%s_list' % set_name
        self.image_list = [line.strip()[:-4] for line in open(params[list_name], 'r')]
        self.num_data = len(self.image_list)
        self.idx = range(self.num_data)
        self.data, self.label = self._read(0)
        self.reset()


    def _read(self, i):
        """ get two list, each list contains two elements:
            name and nd.array value
        """
        data, label = self._read_img(self.image_list[self.idx[i]])
        return list(data.items()), list(label.items())


    def _read_img(self, image_name):
        data_dict = OrderedDict({})
        for i, data_name in enumerate(self.data_name):
            file_name = self.data_setting[data_name]['path'] + image_name + \
                    self.data_setting[data_name]['ext']
            output = self.data_setting[data_name]['reader'](file_name)
            params = self.data_setting[data_name]['trans_params']
            data_dict[data_name] = self.data_setting[data_name]['transform'](
                    output, **params)

        label_dict = OrderedDict({})
        for i, label_name in enumerate(self.label_name):
            file_name = self.label_setting[label_name]['path'] + image_name + \
                    self.label_setting[label_name]['ext']
            output = self.label_setting[label_name]['reader'](file_name)
            params = self.label_setting[label_name]['trans_params']
            label_dict[label_name] = self.label_setting[label_name]['transform'](
                    output, **params)

        return data_dict, label_dict


    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(key, tuple([self.batch_size] + list(value.shape[1:])))
                for key, value in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(key, tuple([self.batch_size] + list(value.shape[1:])))
                for key, value in self.label]

    def get_batch_size(self):
        return self.batch_size

    def reset(self):
        np.random.shuffle(self.idx)
        self.cursor = -1 * self.batch_size

    def iter_next(self):
        if(self.cursor < self.num_data - self.batch_size):
            self.cursor += self.batch_size
            self.cursor = min(self.cursor, self.num_data - self.batch_size)
            return True
        else:
            return False

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            batch_data = [self.array(info[1]) for info in self.provide_data]
            batch_label = [self.array(info[1]) for info in self.provide_label]
            end_curpose = min(self.cursor + self.batch_size, self.num_data)
            pose = 0
            for i in xrange(self.cursor, end_curpose):
                self.data, self.label = self._read(i)
                for info_idx, (key, value) in enumerate(self.data):
                    batch_data[info_idx][pose] = value[0]

                for info_idx, (key, value) in enumerate(self.label):
                    batch_label[info_idx][pose] = value[0]

                pose += 1

            return DataBatch(data=batch_data,
                    label=batch_label,
                    pad=self.batch_size-pose)

        else:
            raise StopIteration



class DensePoseIter(DataIter):
    def __init__(self, params, env,
            setting,
            sampler=None,
            data_names=None,
            label_names=None):

        super(DensePoseIter, self).__init__()
        self.batch_size = params['batch_size']
        self.params = params
        self.is_random = True
        self.sampler = sampler
        self.setting = setting

        self.env = env
        self.data_name = env.state_names if data_names is None else data_names
        self.label_name = env.action_names if label_names is None else label_names

        self.label_num = len(self.label_name)
        self.num_data = params['car_inst_num'] if env.d_name == 'apollo' \
                else params['car_inst_num']

        self.data, self.label = self._read()
        self.reset()


    def _read(self):
        """ get two list, each list contains two elements:
            name and nd.array value
        """
        if self.sampler:
            state, act = self.sampler()
        else:
            state, act = self.env.sample_mcmc_state()
            # reward, _, _, _ = self.env.step(act, is_plot=True)

        data, label = self._read_img(state, act)
        return list(data.items()), list(label.items())


    def _read_img(self, state, act):
        all_in = state.copy()
        data_dict = OrderedDict({})

        for i, data_name in enumerate(self.data_name):
            transform = self.setting[data_name]['transform']
            if 'params' in self.setting[data_name].keys():
                params = self.setting[data_name]['params']
            else:
                if 'depth' in data_name:
                    params = {'mean_depth': state['pose'][0, -1]}
                else:
                    params = {}
            data_dict[data_name] = transform(all_in[data_name], **params)

        label_dict = OrderedDict({})
        for i, label_name in enumerate(self.label_name):
            label_dict[label_name] = act[label_name]

        return data_dict, label_dict


    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(key, tuple([self.batch_size] + list(value.shape[1:])))
                for key, value in self.data]


    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(key, tuple([self.batch_size] + list(value.shape[1:])))
                for key, value in self.label]


    def get_batch_size(self):
        return self.batch_size


    def get_crops(self):
        return self.crops


    def reset(self):
        self.cursor = -1 * self.batch_size


    def iter_next(self):
        return True


    def next(self):
        """return one dict which contains "data" and "label" """

        self.cursor += self.batch_size
        batch_data = [mx.nd.empty(info[1]) for info in self.provide_data]
        batch_label = [mx.nd.empty(info[1]) for info in self.provide_label]
        end_curpose = self.cursor + self.batch_size
        pose = 0

        for i in xrange(self.cursor, end_curpose):
            self.data, self.label = self._read()
            if self.env.counter == self.env.image_num and \
                self.env.inst_counter == len(self.env.valid_id):
                    raise StopIteration

            for info_idx, (key, value) in enumerate(self.data):
                batch_data[info_idx][pose] = value[0]

            for info_idx, (key, value) in enumerate(self.label):
                batch_label[info_idx][pose] = value[0]

            pose += 1

        return DataBatch(data=batch_data,
                label=batch_label,
                pad=self.batch_size-pose)
