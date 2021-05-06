import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import initializers

INF = float("Inf")

class Object(object):
    """
        Generic object
    """
    pass

class SumTree:
    def __init__(self, capacity, e, a):
        self.capacity = capacity
        self.write = 0
        self.trees = {}
        self.main_tree = 0
        self.data = np.repeat(((None, None, None), ), capacity, axis=0)
        self.updates = {}
        self.e = e
        self.a = a
    
    def copy_tree(self, trg_i, src_i=0):
        """Copies src_i's priorities into a new tree trg_i

        Arguments:
            trg_i {object} -- Target tree identifier

        Keyword Arguments:
            src_i {object} -- Source tree identifier (default: {MAIN_TREE})
        """

        if trg_i not in self.trees:
            self.trees[trg_i] = np.copy(self.trees[src_i])
            self.updates[trg_i] = 0

    def create(self, i):
        """Create tree i, either by copying the main tree if it exists, or by
            creating a new tree from scratch

        Arguments:
            i {object} -- The new tree's identifier
        """

        if i is None:
            i = self.main_tree
        if i not in self.trees and self.main_tree in self.trees:
            self.copy_tree(i, self.main_tree)
        elif i not in self.trees:
            self.trees[i] = np.zeros(2 * self.capacity - 1)
            self.updates[i] = 0

    def _propagate(self, idx, change, tree_id=None):
        """Propagate priority changes to root

        Arguments:
            idx {int} -- Node to propagate from
            change {float} -- Priority difference to propagate

        Keyword Arguments:
            tree_id {object} -- Which tree the change applies to (default: {None})
        """

        tree_id = tree_id if tree_id is not None else self.main_tree

        parent = (idx - 1) // 2

        self.trees[tree_id][parent] += change

        if parent != 0:
            self._propagate(parent, change, tree_id)

    def _retrieve(self, idx, s, tree_id=None):
        """Retrieve the node covering offset s starting from node idx

        Arguments:
            idx {int} -- Note to start from
            s {float} -- offset

        Keyword Arguments:
            tree_id {object} -- Which tree the priorities relate to (default: {None})

        Returns:
            int -- node covering the offset
        """

        tree_id = tree_id if tree_id is not None else self.main_tree

        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.trees[tree_id]):
            return idx

        if s <= self.trees[tree_id][left]:
            return self._retrieve(left, s, tree_id)
        else:
            return self._retrieve(right, s - self.trees[tree_id][left],
                                  tree_id)

    def total(self, tree_id=None):
        """Returns the tree's total priority

        Keyword Arguments:
            tree_id {object} -- Tree identifier (default: {None})

        Returns:
            float -- Total priority
        """

        tree_id = tree_id if tree_id is not None else self.main_tree

        return self.trees[tree_id][0]

    def average(self, tree_id=None):
        """Return the tree's average priority, assumes the tree is full

        Keyword Arguments:
            tree_id {object} -- Tree identifier (default: {None})

        Returns:
            float -- Average priority
        """

        return self.total(tree_id) / self.capacity

    def add(self, priorities, data, write=None):
        """Adds a data sample to the SumTree at position write

        Arguments:
            priorities {dict} -- Dictionary of priorities, one key per tree
            data {tuple} -- Transition to be added

        Keyword Arguments:
            write {int} -- Position to write to (default: {None})

        Returns:
            tuple -- Tuple containing the replaced data as well as the node's
                        index in the tree
        """

        write = write if write is not None else self.write
        idx = write + self.capacity - 1

        # Save replaced data to eventually save in secondary memory
        replaced_data = np.copy(self.data[write])
        replaced_priorities = {
            tree: np.copy(self.trees[tree][idx])
            for tree in self.trees
        }
        replaced = (replaced_data, replaced_priorities)

        # Set new priorities
        for i, p in priorities.items():
            self.update(idx, p, i)

        # Set new data
        self.data[write] = data
        return replaced, idx

    def update(self, idx, p, tree_id=None):
        """For a given index, update priorities for the given trees

        Arguments:
            idx {int} -- Node's position in the tree
            p {dict|float} -- Dictionary of priorities or priority for the given tree_id

        Keyword Arguments:
            tree_id {object} -- Tree to be updated (default: {None})
        """

        if type(p) == dict:
            for k in p:
                self.update(idx, p[k], k)
            return
        tree_id = tree_id if tree_id is not None else self.main_tree

        change = p - self.trees[tree_id][idx]

        self.trees[tree_id][idx] = p
        self._propagate(idx, change, tree_id)

    def get(self, s, tree_id=None):
        """Get the node covering the given offset

        Arguments:
            s {float} -- Offset to retrieve

        Keyword Arguments:
            tree_id {object} -- Tree to retrieve from (default: {None})

        Returns:
            tuple -- Containing the index, the priority and the transition
        """

        tree_id = tree_id if tree_id is not None else self.main_tree

        idx = self._retrieve(0, s, tree_id)

        return self.get_by_id(idx, tree_id)

    def get_by_id(self, idx, tree_id=None):
        tree_id = tree_id if tree_id is not None else self.main_tree
        dataIdx = idx - self.capacity + 1

        return (idx, self.trees[tree_id][idx], self.data[dataIdx])

class QueueBuffer:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.repeat(((None, None, None), ), capacity, axis=0)
        # data (transition) transition (trace_id, sample, pred_idx) sample (state, action, reward, next_state, terminal, extra)
        self.write = 0
    
    def add(self, data, write=None):
        write = write if write is not None else self.write

        replaced_data = np.copy(self.data[write])
        self.data[write] = data
        return replaced_data, write

    def get(self, s):
        return self.data[s]


class PrioritizedDiverseMemory:
    def __init__(
            self,
            main_capacity,
            sec_capacity,
            trace_diversity=True,
            crowding_diversity=True,
            value_function=lambda trace, trace_id, memory_indices: np.random.random(1),
            e=0.01,
            a=2):

        self.len = 0
        self.trace_diversity = trace_diversity
        self.value_function = value_function
        self.crowding_diversity = crowding_diversity
        self.e = e
        self.a = a
        self.capacity = main_capacity + sec_capacity
        self.tree = SumTree(self.capacity, e, a)
        self.main_capacity = main_capacity
        self.sec_capacity = sec_capacity
        self.secondary_traces = []
    
    def extract_trace(self, start):
        """Determines the end of the trace starting at position start

        Arguments:
            start {int} -- Trace's starting position

        Returns:
            int -- The trace's end posiition
        """
        trace_id = self.tree.data[start][0]

        end = (start + 1) % self.main_capacity

        if not self.trace_diversity:
            return end
        if trace_id is not None:
            while self.tree.data[end][0] == trace_id:
                end = (end + 1) % self.main_capacity
                if end == start:
                    break

        return end
    
    def remove_trace(self, trace):
        """Removes the trace from the main memory

        Arguments:
            trace {list} -- List of indices for the trace
        """

        _, trace_idx = trace
        for i in trace_idx:
            self.tree.data[i] = (None, None, None)

            idx = i + self.tree.capacity - 1
            for tree in self.tree.trees:
                self.tree.update(idx, 0, tree)

    def get_trace_value(self, trace_tuple):
        """Applies the value_function to the trace's data to compute its value

        Arguments:
            trace_tuple {tuple} -- Tuple containing the trace and the trace's indices

        Returns:
            np.array -- The trace's value
        """

        trace, write_indices = trace_tuple
        if not self.trace_diversity:
            assert len(trace) == 1
        trace_id = trace[0][0]
        trace_data = [t[1] for t in trace]

        return self.value_function(trace_data, trace_id, write_indices)

    def sec_distances(self, traces):
        """Give a set of traces, this method computes each trace's crowding distance

        Arguments:
            traces {list} -- List of trace tuples

        Returns:
            list -- List of distances
        """

        values = [self.get_trace_value(tr) for tr in traces]
        if self.crowding_diversity:
            distances = self.crowd_dist(values)
        else:
            distances = values
        return [(i, d) for i, d in enumerate(distances)], values

    def get_sec_write(self, secondary_traces, trace, reserved_idx=None):
        """
            Given a trace, find free spots in the secondary memory to store it
            by recursively removing past traces with a low crowding distance
        """

        if reserved_idx is None:
            reserved_idx = []
            
        if len(trace) > self.sec_capacity:
            return None
            
        if len(reserved_idx) >= len(trace):
            return reserved_idx[:len(trace)]

        # Find free spots in the secondary memory
        # TODO: keep track of free spots so recomputation isn't necessary
        free_spots = [
            i + self.main_capacity for i in range(self.sec_capacity)
            if (self.tree.data[self.main_capacity + i][1]) is None
        ]
        
        if len(free_spots) > len(reserved_idx):
            return self.get_sec_write(secondary_traces, trace,
                                      free_spots[:len(trace)])

        # Get crowding distance of traces stored in the secondary buffer
        idx_dist, _ = self.sec_distances(secondary_traces)

        # Highest density = lowest distance
        i, _ = min(idx_dist, key=lambda d: d[1])

        _, trace_idx = secondary_traces[i]
        reserved_idx += trace_idx

        self.remove_trace(secondary_traces[i])

        del secondary_traces[i]
        return self.get_sec_write(secondary_traces, trace, reserved_idx)

    def move_to_sec(self, start, end):
        """Move the trace spanning from start to end to the secondary replay
        buffer

        Arguments:
            start {int} -- Start position of the trace
            end {int} -- End position of the trace
        """

        # Recover trace that needs to be moved
        if end <= start:
            indices = np.r_[start:self.main_capacity, 0:end]
        else:
            indices = np.r_[start:end]
        if not self.trace_diversity:
            assert len(indices) == 1
        trace = np.copy(self.tree.data[indices])
        priorities = {
            tree_id: self.tree.trees[tree_id][indices + self.tree.capacity - 1]
            for tree_id in self.tree.trees
        }

        # Get destination indices in secondary memory
        write_indices = self.get_sec_write(self.secondary_traces, trace)

        # Move trace to secondary memory if enough space was freed
        if write_indices is not None and len(write_indices) >= len(trace):
            for i, (w, t) in enumerate(zip(write_indices, trace)):

                self.tree.data[w] = t

                idx = w + self.tree.capacity - 1
                for tree_id in priorities:
                    p = priorities[tree_id][i]
                    self.tree.update(idx, p, tree_id)

                if i > 0:
                    self.tree.data[w][2] = write_indices[i - 1]
            if not self.trace_diversity:
                assert len(trace) == 1
            self.secondary_traces.append((trace, write_indices))
        # elif self.sec_capacity>0:
        #     print("No space found for trace", trace[0][0],", discarding...",file=sys.stderr)

        # Remove trace from main memory
        self.remove_trace((None, indices))
    
    def main_mem_is_full(self):
        """
            Because of the circular way in which we fill the memory, checking
            whether the current write position is free is sufficient to know
            if the memory is full.
        """
        return self.tree.data[self.tree.write][1] is not None
    
    
    def crowd_dist(self, datas):
        points = np.array([Object() for _ in datas])
        dimensions = len(datas[0])
        for i, d in enumerate(datas):
            points[i].data = d
            points[i].i = i
            points[i].distance = 0.

        # Compute the distance between neighbors for each dimension and add it to
        # each point's global distance
        for d in range(dimensions):
            points = sorted(points, key=lambda p: p.data[d])
            spread = points[-1].data[d] - points[0].data[d]
            for i, p in enumerate(points):
                if i == 0 or i == len(points) - 1:
                    p.distance += INF
                else:
                    p.distance += (
                        points[i + 1].data[d] - points[i - 1].data[d]) / spread

        # Sort points back to their original order
        points = sorted(points, key=lambda p: p.i)
        distances = np.array([p.distance for p in points])

        return distances
    
    def sample(self, n, tree_id=None):
        """Sample n transitions from the replay buffer, following the priorities
        of the tree identified by tree_id

        Arguments:
            n {int} -- Number of transitios to sample

        Keyword Arguments:
            tree_id {object} -- identifier of the tree whose priorities should be followed (default: {None})

        Returns:
            tuple -- pair of (indices, transitions)
        """
        if n<1:
            return None, None, None

        batch = np.zeros((n, ), dtype=np.ndarray)
        ids = np.zeros(n, dtype=int)
        priorities = np.zeros(n, dtype=float)
        segment = self.tree.total(tree_id) / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            (idx, p, data) = self.tree.get(s, tree_id)
            while (data[1]) is None or (
                    idx - self.capacity + 1 >= self.capacity):
                s = np.random.uniform(0, self.tree.total(tree_id))
                (idx, p, data) = self.tree.get(s, tree_id)
            ids[i] = idx
            batch[i] = data[1]
            priorities[i] = p
        return ids, batch, priorities
    
    def add(self, error, sample, trace_id=None, pred_idx=None, tree_id=None):
        self.len = min(self.len + 1, self.capacity)
        self.tree.create(tree_id)

        sample = (trace_id, sample, None
                  if pred_idx is None else (pred_idx - self.capacity + 1))

        # Free up space in main memory if necessary
        if self.main_mem_is_full():
            end = self.extract_trace(self.tree.write)
            self.move_to_sec(self.tree.write, end)

        # Save sample into main memory
        if type(error) is not dict:
            error = {tree_id: error}
        idx = self.add_sample(sample, error, self.tree.write)
        self.tree.write = (self.tree.write + 1) % self.main_capacity

        return idx
    
    def add_sample(self, transition, error, write=None):
        """Stores the transition into the priority tree
        """

        p = {k: self._getPriority(error[k]) for k in error}
        _, idx = self.tree.add(p, transition, write=write)
        return idx
    

    def get(self, indices):
        """Given a list of node indices, this method returns the data stored at
        those indices

        Arguments:
            indices {list} -- List of indices

        Returns:
            np.array -- array of transitions
        """

        indices = np.array(indices, dtype=int) - self.capacity + 1
        return self.tree.data[indices][:, 1]

    def update(self, idx, error, tree_id=None):
        """Given a node's idx, this method updates the corresponding priority in
        the tree identified by tree_id

        Arguments:
            idx {int} -- Node's index
            error {float} -- New error

        Keyword Arguments:
            tree_id {object} -- Identifies the tree to update (default: {None})
        """

        if tree_id is None:
            tree_id = self.tree.main_tree
        if type(error) is not dict:
            error = {tree_id: error}
        p = {k: self._getPriority(error[k]) for k in error}
        self.tree.update(idx, p, tree_id)
    
    def _getPriority(self, error):
        """Compute priority from error

        Arguments:
            error {float} -- error

        Returns:
            float -- priority
        """

        return (error + self.e)**self.a

    def _getError(self, priority):
        """Given a priority, computes the corresponding error

        Arguments:
            priority {float} -- priority

        Returns:
            float -- error
        """

        return priority**(1 / self.a) - self.e

    def get_error(self, idx, tree_id=None):
        tree_id = self.tree.main_tree if tree_id is None else tree_id
        priority = self.tree.trees[tree_id][idx]
        return self._getError(priority)
    
    def add_tree(self, tree_id):
        """Adds a secondary priority tree

        Arguments:
            tree_id {object} -- The secondary tree's id
        """

        self.tree.create(tree_id)

    def get_data(self, include_indices=False):
        """Get all the data stored in the replay buffer 

        Keyword Arguments:
            include_indices {bool} -- Whether to include each sample's position in the replay buffer (default: {False})

        Returns:
            The data
        """

        all_data = list(np.arange(self.capacity) + self.capacity - 1), list(
            self.tree.data)
        indices = []
        data = []
        for i, d in zip(all_data[0], all_data[1]):
            if (d[1]) is not None:
                indices.append(i)
                data.append(d[1])
        if include_indices:
            return indices, data
        else:
            return data

    

class MemoryBuffer(PrioritizedDiverseMemory):

    def __init__(
            self,
            main_capacity,
            sec_capacity,
            trace_diversity=True,
            crowding_diversity=True,
            value_function=lambda trace, trace_id, memory_indices: np.random.random(1),
            e=0.01,
            a=2):

        # super(MemoryBuffer, self).__init__(main_capacity, sec_capacity, trace_diversity, crowding_diversity, value_function, e, a)
        PrioritizedDiverseMemory.__init__(self, main_capacity, sec_capacity, trace_diversity, crowding_diversity, value_function, e, a)
    

class AttentiveMemoryBuffer(PrioritizedDiverseMemory):
    def __init__(
            self,
            main_capacity,
            sec_capacity,
            trace_diversity=True,
            crowding_diversity=True,
            value_function=lambda trace, trace_id, memory_indices: np.random.random(1),
            e=0.01,
            a=2):

        # super(AttentiveMemoryBuffer, self).__init__(main_capacity, sec_capacity, trace_diversity, crowding_diversity, value_function, e, a)
        PrioritizedDiverseMemory.__init__(self, main_capacity, sec_capacity, trace_diversity, crowding_diversity, value_function, e, a)

    def cal_weights_similarity(self, a, b):
        dist = np.linalg.norm(a-b)

        return dist
    
    def cal_states_similarity(self, state):
        seed = 0
        input_t = tf.convert_to_tensor(state, dtype=np.float32)
        x = Lambda(lambda x: x / 255., name="input_normalizer")(input_t)

        x = TimeDistributed(Conv2D(filters=32, kernel_size=6, strides=2, 
                                    activation='relu', kernel_initializer=initializers.GlorotUniform(seed),
                                    input_shape=x.shape))(x)
        x = TimeDistributed(MaxPool2D())(x)

        x = TimeDistributed(Conv2D(filters=64, kernel_size=5, strides=2, 
                                    activation='relu', kernel_initializer=initializers.GlorotUniform(seed)))(x)
        x = TimeDistributed(MaxPool2D())(x)

        x = Flatten()(x)
        state = x.numpy()

        dist = np.linalg.norm(state[0, :]-state[1, :])
        
        # print(dist)
        return dist

    def sample(self, n, k, current_weights, current_state, tree_id=None):
        """Sample n transitions from the replay buffer, following the priorities
        of the tree identified by tree_id

        Arguments:
            n {int} -- Number of transitios to sample
            k {int} -- Sample k * n number of transitons, k will anneal from K to 1

        Keyword Arguments:
            tree_id {object} -- identifier of the tree whose priorities should be followed (default: {None})

        Returns:
            tuple -- pair of (indices, transitions)
        """
        if n<1:
            return None, None, None

        batch = np.zeros((n, ), dtype=np.ndarray)
        ids = np.zeros(n, dtype=int)
        priorities = np.zeros(n, dtype=float)
        segment = self.tree.total(tree_id) / n
        score = dict()

        for i in range(int(round(n*k))):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            (idx, p, data) = self.tree.get(s, tree_id)
            while (data[1]) is None or (
                    idx - self.capacity + 1 >= self.capacity):
                s = np.random.uniform(0, self.tree.total(tree_id))
                (idx, p, data) = self.tree.get(s, tree_id)
            
            state = np.concatenate((np.expand_dims(data[1][0], axis=0), 
                                        np.expand_dims(current_state, axis=0)))
            sim_score = self.cal_weights_similarity(current_weights, data[1][5][3]) + self.cal_states_similarity(state)
            score[i] = (idx, sim_score, data, p) #data[1][5][3] means the weights used in this transition
        
        score = sorted(score.items(), key=lambda item: item[1][1], reverse=False)
        for i in range(n):
            ids[i] = score[i][1][0]
            batch[i] = score[i][1][2][1]
            priorities[i] = score[i][1][3]
        
        return ids, batch, priorities

    # def sample(self, n, k, current_weights, tree_id=None):
    #     """Sample n transitions from the replay buffer, following the priorities
    #     of the tree identified by tree_id

    #     Arguments:
    #         n {int} -- Number of transitios to sample
    #         k {int} -- Sample k * n number of transitons, k will anneal from K to 1

    #     Keyword Arguments:
    #         tree_id {object} -- identifier of the tree whose priorities should be followed (default: {None})

    #     Returns:
    #         tuple -- pair of (indices, transitions)
    #     """
    #     if n<1:
    #         return None, None, None

    #     batch = np.zeros((n, ), dtype=np.ndarray)
    #     ids = np.zeros(n, dtype=int)
    #     priorities = np.zeros(n, dtype=float)
    #     segment = self.tree.total(tree_id) / n
    #     score = dict()

    #     for i in range(int(round(n*k))):
    #         a = segment * i
    #         b = segment * (i + 1)

    #         s = np.random.uniform(a, b)
    #         (idx, p, data) = self.tree.get(s, tree_id)
    #         while (data[1]) is None or (
    #                 idx - self.capacity + 1 >= self.capacity):
    #             s = np.random.uniform(0, self.tree.total(tree_id))
    #             (idx, p, data) = self.tree.get(s, tree_id)
            
    #         score[i] = (idx, self.cal_similarity(current_weights, data[1][5][3]), data, p) #data[1][5][3] means the weights used in this transition
        
    #     score = sorted(score.items(), key=lambda item: item[1][1], reverse=False)
    #     for i in range(n):
    #         ids[i] = score[i][1][0]
    #         batch[i] = score[i][1][2][1]
    #         priorities[i] = score[i][1][3]
        
    #     return ids, batch, priorities

    


class DiverseMemoryWithAER():

    def __init__(self,
            main_capacity,
            sec_capacity,
            trace_diversity=True,
            crowding_diversity=True,
            value_function=lambda trace, trace_id, memory_indices: np.random.random(1)
            ):

        self.len = 0
        self.trace_diversity = trace_diversity
        self.value_function = value_function
        self.capacity = main_capacity + sec_capacity
        self.buffer = QueueBuffer(self.capacity)
        self.crowding_diversity = crowding_diversity
        self.main_capacity = main_capacity
        self.sec_capacity = sec_capacity
        self.secondary_traces = []
    
    def sample_attentive(self, n, k, current_state):
        if n < 1:
            return None, None, None

        batch = np.zeros((n, ), dtype=np.ndarray)
        ids = np.zeros(n, dtype=int)
        score = dict()
        for i in range(int(round(n*k))):
            id = np.random.randint(0, self.capacity)
            while self.buffer.data[id][1] is None:
                id = np.random.randint(0, self.capacity)
            state = np.concatenate((np.expand_dims(self.buffer.data[id][1][0], axis=0), 
                                        np.expand_dims(current_state, axis=0)))
            score[id] = self.cal_similarity(state)

        score = sorted(score.items(), key=lambda item: item[1], reverse=False)
        for i in range(n):
            ids[i] = score[i][0]
            batch[i] = self.buffer.data[ids[i]][1]
            priorities = None
        return ids, batch, priorities

    def cal_similarity(self, state):

        input_t = tf.convert_to_tensor(state, dtype=np.float32)
        x = Lambda(lambda x: x / 255., name="input_normalizer")(input_t)

        x = TimeDistributed(Conv2D(filters=32, kernel_size=6, strides=2, 
                                    activation='relu', kernel_initializer='glorot_uniform',
                                    input_shape=x.shape))(x)
        x = TimeDistributed(MaxPool2D())(x)

        x = TimeDistributed(Conv2D(filters=48, kernel_size=5, strides=2, 
                                    activation='relu', kernel_initializer='glorot_uniform'))(x)
        x = TimeDistributed(MaxPool2D())(x)

        x = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(x)

        x = Flatten()(x)
        state = x.numpy()

        dist = np.linalg.norm(state[0, :]-state[1, :])
        
        # print(dist)
        return dist

    def sample(self, n):
        if n < 1:
            return None, None, None
        
        batch = np.zeros((n, ), dtype=np.ndarray)
        ids = np.zeros(n, dtype=int)
        for i in range(n):
            id = np.random.randint(0, self.capacity)
            while self.buffer.data[id][1] is None:
                id = np.random.randint(0, self.capacity)
            ids[i] = id
            batch[i] = self.buffer.data[ids[i]][1]
            priorities = None
        return ids, batch, priorities

    def add(self, sample, trace_id=None, pred_idx=None):
        self.len = min(self.len + 1, self.capacity)
        
        sample = (trace_id, sample, pred_idx)
        
        if self.main_mem_is_full():
            end = self.extract_trace(self.buffer.write)
            self.move_to_sec(self.buffer.write, end)
        
        idx = self.add_sample(sample, self.buffer.write)
        self.buffer.write = (self.buffer.write + 1) % self.main_capacity

        return idx
    
    def add_sample(self, transition, write=None):
        
        _, idx = self.buffer.add(transition, write=write)

        return idx

    def get_data(self, include_indices=False):
        all_data = list(np.arange(self.capacity)), list(self.buffer.data)
        indices = []
        data = []
        for i, d in zip(all_data[0], all_data[1]):
            if (d[1]) is not None:
                indices.append(i)
                data.append(d[1])
            if include_indices:
                return indices, data
            else:
                return data

    def get(self, indices):

        indices = np.array(indices, dtype=int)

        return self.buffer.data[indices][:, 1]

    def main_mem_is_full(self):
        
        return self.buffer.data[self.buffer.write][1] is not None
    
    def extract_trace(self, start):

        trace_id = self.buffer.data[start][0]

        end = (start+1) % self.main_capacity

        if not self.trace_diversity:
            return end
        if trace_id is not None:
            while self.buffer.data[end][0] == trace_id:
                end = (end + 1) % self.main_capacity
                if end == start:
                    break
        return end

    def move_to_sec(self, start, end):
        if end <= start:
            indices = np.r_[start:self.main_capacity, 0:end]
        else:
            indices = np.r_[start:end]
        
        if not self.trace_diversity:
            assert len(indices) == 1

        trace = np.copy(self.buffer.data[indices])
        write_indices = self.get_sec_write(self.secondary_traces, trace)

        if write_indices is not None and len(write_indices) >= len(trace):
            for i, (w, t) in enumerate(zip(write_indices, trace)):

                self.buffer.data[w] = t
                if i > 0:
                    self.buffer.data[w][2] = write_indices[i - 1]
            
            if not self.trace_diversity:
                assert len(trace) == 1
            self.secondary_traces.append((trace, write_indices))
        
        self.remove_trace((None, indices))
    
    def remove_trace(self, trace):
        _, trace_idx = trace
        for i in trace_idx:
            self.buffer.data[i] = (None, None, None)

    def get_sec_write(self, secondary_traces, trace, reserved_idx=None):

        if reserved_idx is None:
            reserved_idx = []
            
        if len(trace) > self.sec_capacity:
            return None
            
        if len(reserved_idx) >= len(trace):
            return reserved_idx[:len(trace)]

        # Find free spots in the secondary memory
        # TODO: keep track of free spots so recomputation isn't necessary
        free_spots = [
            i + self.main_capacity for i in range(self.sec_capacity)
            if (self.buffer.data[self.main_capacity + i][1]) is None
        ]
        
        if len(free_spots) > len(reserved_idx):
            return self.get_sec_write(secondary_traces, trace,
                                      free_spots[:len(trace)])

        # Get crowding distance of traces stored in the secondary buffer
        idx_dist, _ = self.sec_distances(secondary_traces)

        # Highest density = lowest distance
        i, _ = min(idx_dist, key=lambda d: d[1])

        _, trace_idx = secondary_traces[i]
        reserved_idx += trace_idx

        self.remove_trace(secondary_traces[i])

        del secondary_traces[i]
        return self.get_sec_write(secondary_traces, trace, reserved_idx)

    def sec_distances(self, traces):
        values = [self.get_trace_value(tr) for tr in traces]
        if self.crowding_diversity:
            distances = self.crowd_dist(values)
        else:
            distances = values
        return [(i, d) for i, d in enumerate(distances)], values

    def get_trace_value(self, trace_tuple):
        trace, write_indices = trace_tuple
        if not self.trace_diversity:
            assert len(trace) == 1
        trace_id = trace[0][0]
        trace_data = [t[1] for t in trace]

        return self.value_function(trace_data, trace_id, write_indices)
    
    def crowd_dist(self, datas):
        points = np.array([Object() for _ in datas])
        dimensions = len(datas[0])
        for i, d in enumerate(datas):
            points[i].data = d
            points[i].i = i
            points[i].distance = 0.

        # Compute the distance between neighbors for each dimension and add it to
        # each point's global distance
        for d in range(dimensions):
            points = sorted(points, key=lambda p: p.data[d])
            spread = points[-1].data[d] - points[0].data[d]
            for i, p in enumerate(points):
                if i == 0 or i == len(points) - 1:
                    p.distance += INF
                else:
                    p.distance += (
                        points[i + 1].data[d] - points[i - 1].data[d]) / spread

        # Sort points back to their original order
        points = sorted(points, key=lambda p: p.i)
        distances = np.array([p.distance for p in points])

        return distances