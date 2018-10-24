import pdb
import cv2

import mxnet as mx
import numpy as np
from imgaug import augmenters as iaa
from mxnet.io import DataIter, DataBatch
import utils.transforms as trs
import utils.utils as uts

from collections import OrderedDict, namedtuple
Batch = namedtuple('Batch', ['data'])


class EnvDataIter(DataIter):
    """
    Parameters
    ----------

    """
    def __init__(self,
                 params,
                 env,
                 agent):

        super(EnvDataIter, self).__init__()
        self.batch_size = agent.batch_size
        self.is_random = True
        self.env = env
        self.agent = agent

        self.data_name = env.state_names + agent.action_names
        self.label_name = env.reward_names
        self.transforms = OrderedDict(
               env.transforms.items() + agent.transforms.items())

        self.label_num = len(self.label_name)
        self.num_data = params['car_inst_num']

        self.data, self.label = self._read()
        self.reset()


    def _read(self):
        """ get two list, each list contains two elements:
            name and nd.array value
        """
        state = self.env.sample_state()
        action = self.agent.random_action()
        reward, _, _, _ = self.env.step(action)
        data, label = self._read_img(state, action, reward)

        return list(data.items()), list(label.items())


    def _read_img(self, state, action, reward):
        all_in = state
        all_in.update(action)
        all_in.update(reward)
        data_dict = OrderedDict({})

        for i, data_name in enumerate(self.data_name):
            if not ('pose' in data_name):
                data_dict[data_name] = \
                        self.transforms[data_name]['transform'](
                        np.float32(all_in[data_name]))
            else:
                data_dict[data_name] = np.float32(all_in[data_name])

        label_dict = OrderedDict({})
        for i, label_name in enumerate(self.label_name):
            label_dict[label_name] = np.float32(all_in[label_name])

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
            batch_data = [mx.nd.empty(info[1]) for info in self.provide_data]
            batch_label = [mx.nd.empty(info[1]) for info in self.provide_label]
            end_curpose = min(self.cursor + self.batch_size, self.num_data)
            pose = 0
            for i in xrange(self.cursor, end_curpose):
                self.data, self.label = self._read()
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


class EnvPolicyDataIter(DataIter):
    """
    Parameters
    ----------

    """
    def __init__(self,
                 params,
                 env,
                 data_names=None,
                 label_names=None,
                 is_crop=False):

        super(EnvPolicyDataIter, self).__init__()
        self.batch_size = params['batch_size']
        self.params = params
        self.is_random = True
        self.env = env
        self.crop = is_crop

        self.data_name = env.state_names if data_names is None \
                else data_names
        self.label_name = env.reward_names if label_names is None \
                else label_names

        self.label_num = len(self.label_name)
        self.num_data = params['car_inst_num']

        self.data, self.label, _ = self._read()
        self.reset()


    def _read(self):
        """ get two list, each list contains two elements:
            name and nd.array value
        """
        state = self.env.sample_state()
        data, label, crop = self._read_img(state)

        return list(data.items()), list(label.items()), crop


    def _read_img(self, state):
        all_in = state.copy()
        data_dict = OrderedDict({})

        # crop image
        crop = []
        if self.crop:
            crop = uts.get_mask_bounding_box(state['mask'], context=1.0)
            for i, data_name in enumerate(self.data_name):
                if self.env.transforms[data_name]['is_image']:
                    all_in[data_name] = uts.crop_image(
                            all_in[data_name], crop)
                    inter = self.env.transforms[data_name]['interpolation']
                    all_in[data_name] = cv2.resize(
                            np.float32(all_in[data_name]),
                            tuple(self.params['crop_size'][::-1]),
                            interpolation=inter)

        for i, data_name in enumerate(self.data_name):
            data_dict[data_name] = self.env.transforms[data_name]['transform'](
                    np.float32(all_in[data_name]))

        label_dict = OrderedDict({})

        return data_dict, label_dict, crop


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
        """return one dict which contains "data" and "label" """

        if(self.cursor < self.num_data - self.batch_size):
            self.cursor += self.batch_size
            self.cursor = min(self.cursor, self.num_data - self.batch_size)
            return True
        else:
            return False


    def next(self):
        """return one dict which contains "data" and "label" """

        if self.iter_next():
            batch_data = [mx.nd.empty(info[1]) for info in self.provide_data]
            batch_label = [mx.nd.empty(info[1]) for info in self.provide_label]
            end_curpose = min(self.cursor + self.batch_size, self.num_data)
            pose = 0
            self.crops = np.zeros((self.batch_size, 4))

            for i in xrange(self.cursor, end_curpose):
                self.data, self.label, crop = self._read()
                # print self.data['pose'], self.label['del_pose']

                if self.crop:
                    self.crops[pose, :] = crop

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


class PolicyDataIter(DataIter):
    def __init__(self, params, env,
            setting,
            sampler=None,
            data_names=None,
            label_names=None):

        super(PolicyDataIter, self).__init__()
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




