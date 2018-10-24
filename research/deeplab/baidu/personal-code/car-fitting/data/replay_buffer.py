from collections import deque
import heapq
from Queue import PriorityQueue
import numpy as np
import random
from collections import OrderedDict
import pdb


class ReplayBuffer(object):

    def __init__(self, buffer_size,
            state_names, action_names,
            update_state_names=None, verbose=False):
        self.buffer_size = buffer_size
        self.state_names = state_names
        self.action_names = action_names
        self.verbose = verbose

        # some state are static during the game
        self.update_state_names = update_state_names \
                if update_state_names else {}
        self.num_experiences = 0
        self.buffer = deque()


    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        sample = random.sample(self.buffer, batch_size)

        state_batch = OrderedDict({})
        state_num = len(self.state_names)
        for i, state_name in enumerate(self.state_names):
            state_batch[state_name] = np.concatenate(
                    [data[i] for data in sample], axis=0)

        action_batch = OrderedDict({})
        act_num = len(self.action_names)
        for i, action_name in enumerate(self.action_names):
            act_id = i + state_num
            action_batch[action_name] = np.concatenate(
                    [data[act_id] for data in sample], axis=0)

        reward_batch = OrderedDict({})
        reward_id = act_num + state_num
        reward_batch['reward'] = np.concatenate(
                [data[reward_id] for data in sample], axis=0)

        next_state = OrderedDict({})
        update_state_num = len(self.update_state_names)
        for i, name in enumerate(self.update_state_names):
            idx = reward_id + i + 1
            next_state[name] = np.concatenate(
                    [data[idx] for data in sample], axis=0)

        done_id = reward_id + 1 + update_state_num
        done_batch = np.array([data[done_id] for data in sample])

        return state_batch, action_batch, reward_batch, \
                next_state, done_batch


    def size(self):
        return self.buffer_size


    def add(self, state, action, reward, new_state, done):
        assert bool(state), 'state can not be empty'
        assert bool(action)
        assert bool(reward)
        experience = state.values() + action.values() + reward.values() + \
                new_state.values() + [done]

        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


class PriorityReplayBuffer(ReplayBuffer):

    def __init__(self, buffer_size,
            state_names, action_names,
            update_state_names=None):
        self.buffer_size = buffer_size
        self.state_names = state_names
        self.action_names = action_names

        # some state are static during the game
        self.update_state_names = update_state_names \
                if update_state_names else {}
        self.num_experiences = 0
        self.buffer = []

    def _sample_unique(self):
        """ sample from different state
        """
        sample = []
        pass




    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        sample = random.sample(self.buffer, batch_size)

        state_batch = OrderedDict({})
        state_num = len(self.state_names)
        for i, state_name in enumerate(self.state_names):
            state_batch[state_name] = np.concatenate(
                    [data[i] for data in sample], axis=0)

        action_batch = OrderedDict({})
        act_num = len(self.action_names)
        for i, action_name in enumerate(self.action_names):
            act_id = i + state_num
            action_batch[action_name] = np.concatenate(
                    [data[act_id] for data in sample], axis=0)

        reward_batch = OrderedDict({})
        reward_id = act_num + state_num
        reward_batch['reward'] = np.concatenate(
                [data[reward_id] for data in sample], axis=0)

        next_state = OrderedDict({})
        update_state_num = len(self.update_state_names)
        for i, name in enumerate(self.update_state_names):
            idx = reward_id + i + 1
            next_state[name] = np.concatenate(
                    [data[idx] for data in sample], axis=0)

        done_id = reward_id + 1 + update_state_num
        done_batch = np.array([data[done_id] for data in sample])

        return state_batch, action_batch, reward_batch, \
                next_state, done_batch


    def add(self, state, action, reward, new_state, done):
        assert bool(state), 'state can not be empty'
        assert bool(action)
        assert bool(reward)
        experience = state.values() + action.values() + reward.values() + \
                new_state.values() + [done]

        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)


    def erase(self):
        self.buffer = heapq()
        self.num_experiences = 0


class InstanceSetBuffer(object):
    """ This buffer memorize the whole dataset
    """
    def __init__(self, state_names, action_names, buffer_size, verbose=False):

        # some state are static during the game
        self.state_names = state_names
        self.action_names = action_names
        self.num_experiences = 0
        self.buffer = OrderedDict({})
        self.verbose = verbose
        self.buffer_size = buffer_size


    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        inst_names = random.sample(self.buffer.keys(), batch_size)

        state_batch = OrderedDict({})
        for i, state_name in enumerate(self.state_names):
            state_batch[state_name] = np.concatenate(
                    [self.buffer[name][state_name] for name in inst_names], axis=0)

        action_batch = OrderedDict({})
        for i, action_name in enumerate(self.action_names):
            action_batch[action_name] = np.concatenate(
                    [self.buffer[name][action_name] for name in inst_names], axis=0)

        reward_batch = OrderedDict({})
        reward_batch['reward'] = np.concatenate(
                [self.buffer[name]['reward'] for name in inst_names], axis=0)

        next_state = {}
        done_batch = []
        return state_batch, action_batch, reward_batch, \
               next_state, done_batch

    def get_a_sample(self, key):
        return {name: self.buffer[key][name] for name in self.state_names}, \
                {name: self.buffer[key][name] for name in self.action_names}, \
                {'reward': self.buffer[key]['reward']}



    def size(self):
        return len(self.buffer.keys())


    def state_to_key(self, state):
        return '%s_%s' % (state['image_name'], state['inst_id'])


    def add(self, state, action, reward, new_state, done):
        assert bool(state), 'state can not be empty'
        assert bool(action)
        assert bool(reward)

        key = self.state_to_key(state)
        temp = state
        temp.update(action)
        temp.update(reward)

        if key in self.buffer.keys():
            if reward['reward'] > self.buffer[key]['reward']:
                self.buffer[key] = temp
        else:
            if len(self.buffer) > self.buffer_size:
                inst_name = random.sample(self.buffer.keys(), 1)[0]
                self.buffer.pop(inst_name)
            self.buffer[key] = temp
            self.num_experiences += 1

        if self.verbose:
            print key, temp['reward'], temp['del_pose']


    def has_state(self, state):
        key = self.state_to_key(state)
        if key in self.buffer:
            return True
        else:
            return False


    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences


if __name__ == '__main__':
    state_names = ['image', 'depth', 'mask', 'crop', 'pose']
    action_names = ['del_pose']
    state = OrderedDict([('image', np.zeros((1, 3, 3, 3))),
        ('depth', np.zeros((1, 3, 3))),
        ('mask', np.zeros((1, 3, 3))),
        ('crop', np.zeros((1, 4))),
        ('pose', np.zeros((1, 6)))])
    act = OrderedDict([('del_pose', np.zeros((1, 6)))])
    reward = {'reward': np.zeros((1, 1))}
    buffer_inst = InstanceSetBuffer(state_names, action_names)
    state['image_name'] = '0'
    state['inst_id'] = '0'
    buffer_inst.add(state, act, reward, {}, False)
    state['image_name'] = '0'
    state['inst_id'] = '1'
    buffer_inst.add(state, act, reward, {}, False)
    state['image_name'] = '0'
    state['inst_id'] = '2'
    buffer_inst.add(state, act, reward, {}, False)

    temp = buffer_inst.get_batch(2)
    for item in temp:
        print item.keys()


