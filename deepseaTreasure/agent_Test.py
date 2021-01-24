from __future__ import print_function

import gc
import math
import os
import random
import time
from optparse import OptionParser

import keras.backend as K
import scipy.spatial
from operator import mul
from keras import initializers
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import LearningRateScheduler
from keras.layers import *
from keras.layers.pooling import *
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import Model, load_model
from keras.optimizers import *
from keras.utils import np_utils

from config_agent import *
from diverse_mem import DiverseMemory
from history import *
from utils import *
from network import *

from pprint import pprint

import gym
from deep_sea_treasure import DeepSeaTreasure

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def generate_weights(count=1, n=3, m=1):
    all_weights = []
    target = np.random.dirichlet(np.ones(n), 1)[0]
    prev_t = target
    for _ in range(count // m):
        target = np.random.dirichlet(np.ones(n), 1)[0]
        if m == 1:
            all_weights.append(target)
        else:
            for i in range(m):
                i_w = target * (i + 1) / float(m) + prev_t * \
                    (m - i - 1) / float(m)
                all_weights.append(i_w)
        prev_t = target + 0.

    return all_weights


try:
    from PIL import Image
except:
    import Image

DEBUG = 1

class DeepAgent():
    def __init__(self,
                 actions,
                 objective_cnt,
                 memory_size=10000,
                 sample_size=64,
                 weights=None,
                 start_discount=0.95,
                 discount=0.95,
                 end_discount=0.95,
                 start_e=0.1,
                 end_e=0.01,
                 learning_rate=1e-1,
                 target_update_interval=150,
                 frame_skip=4,
                 alg="scal",
                 memory_type="STD",
                 mem_a=1,
                 mem_e=0.02,
                 update_interval=4,
                 reuse='full',
                 extra="",
                 clipnorm=0,
                 clipvalue=0,
                 nesterov=True,
                 momentum=0.9,
                 dupe=False,
                 start_annealing=0.01,
                 min_buf_size=0.1,
                 scale=1,
                 im_size=(IM_SIZE, IM_SIZE),
                 grayscale=BLACK_AND_WHITE,
                 frames_per_state=2,
                 max_episode_length=1000,
                 lstm=False):
        """Agent implementing both Multi-Network, Multi-Head and Single-head 
            algorithms

        Arguments:
            actions {list} -- List of possible actions
            objective_cnt {int} -- Number of objectives

        Keyword Arguments:
            memory_size {int} -- Size of the replay buffer (default: {100000})
            sample_size {int} -- Number of samples used for updates (default: {64})
            weights {list} -- Default weights (default: {None})
            discount {float} -- Discount facctor (default: {0.98})
            start_e {float} -- Starting exploration rate (default: {1.})
            end_e {float} -- Final exploration rate (default: {0.05})
            learning_rate {float} -- Learning rate (default: {1e-1})
            target_update_interval {int} -- Interval between target network synchronizations (default: {600})
            frame_skip {int} -- Number of times each action is repeated (default: {4})
            alg {str} -- Algorithm to use, either 'scal','mo','cond' or 'meta' (default: {"scal"})
            memory_type {str} -- Either 'std' for a prioritized experience replay or 'crowd' for a diversity-enforced experience replay (default: {"STD"})
            update_interval {int} -- Interval between network updates (default: {4})
            reuse {str} -- 'full' for full reuse, 'proportional' for proportional reuse or 'sectioned' for no reuse (default: {'full'})
            extra {str} -- Additional information string identifying the learner (default: {""})
            clipnorm {float} -- clipnorm value (losses are normalized to this value) (default: {1})
            clipvalue {float} -- clipvalue (losses are clipped to this value) (default: {1})
            nesterov {bool} -- Wether to use nesterov momentum (default: {True})
            momentum {float} -- Momentum parameter for the SGD optimizer (default: {0.9})
            dupe {bool} -- If True, samples are trained both on the current and on a past weight (default: {False})
            min_buf_size {float} -- Minimum size of the buffer needed to start updating, as a fraction of the number of learning steps (default: {0.1})
            scale {float} -- Scales the network's layer sizes (default: {1})
            im_size {tuple} -- Size the frames should be scaled to (default: {(48, 48)})
            grayscale {bool} -- Color mode for the frames (default: {True})
            frames_per_state {int} -- How many frames the state should consist of (default: {2})
            max_episode_length {int} -- Interrupt episodes if they last longer than this (default: {5000})
        """
        self.max_error = .0
        self.direct_update = True
        self.normalize_weights = False
        self.recent_experiences = []
        self.epsilon = self.start_e = start_e
        self.max_episode_length = max_episode_length
        self.nesterov = nesterov
        self.actions = actions
        self.action_count = len(actions)
        self.frames_per_state = frames_per_state
        self.im_size = im_size
        self.grayscale = grayscale
        self.history = History(frames_per_state, im_size, grayscale)
        self.mem_a = mem_a
        self.mem_e = mem_e
        self.scale = scale
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.momentum = momentum
        self.frame_skip = frame_skip
        self.reuse = reuse
        self.start_e = start_e
        self.end_e = end_e
        self.start_annealing = start_annealing
        self.is_first_update = True
        self.trace_values={}
        self.alg = alg
        self.dupe = dupe
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.min_buf_size = min_buf_size
        self.actions = actions
        self.lstm = lstm

        self.memory_size = memory_size

        self.action_count = len(self.actions)
        self.obj_cnt = objective_cnt

        self.sample_size = sample_size

        self.weight_history = []
        if weights is not None:
            self.set_weights(np.array(weights))

        self.lr = learning_rate
        self.memory_type = memory_type

        self.build_models()
        self.initialize_memory()

        self.start_discount = start_discount
        self.discount = start_discount
        self.end_discount = end_discount

        self.extra = extra
        print(vars(self))

    def build_models(self):
        """Builds the networks then synchronizes the model and target networks
        """

        self.input_shape = (self.frames_per_state,) + self.history.im_shape
        self.network = Network(
                 self.obj_cnt,
                 self.action_count,
                 self.scale, 
                 self.lstm, 
                 self.alg,
                 self.lr,
                 self.clipnorm,
                 self.clipvalue,
                 self.momentum,
                 self.nesterov,
                 self.input_shape)
        self.model, self.target_model, self.trainable_model = self.network.build_networks()

        self.model.summary()
        self.update_target()
   
    def qvalue_dim(self):
        return 1 if self.has_scalar_qvalues() else self.obj_cnt

    def policy_update(self):
        """Update the policy (Q-values)

        Returns:
            float -- The update's loss
        """

        np.random.seed(self.steps)
        ids, batch, _ = self.buffer.sample(self.sample_size)

        if self.direct_update:
            # Add recent experiences to the priority update batch
            batch = np.concatenate(
                (batch, self.buffer.get(self.recent_experiences)), axis=0)
            ids = np.concatenate((ids, self.recent_experiences)).astype(int)

        # if self.duplicate is True, we train each sample in the batch on two
        # weight vectors, hence we duplicate the batch data
        if self.duplicate():
            batch = np.repeat(batch, 2, axis=0)
            ids = np.repeat(ids, 2, axis=0)

        model_Qs, target_Qs, w_batch, states = self.get_training_data(batch)

        # Every other value in model_QS, target_Qs, w_batch and states concerns
        # the subsequent state. In other words, states=(s_i,s_{i+1},s_j,s_{j+1},...)
        x_batch = states[::2]
        y_batch = np.copy(model_Qs[::2])

        # Mask the output to ignore losses of non-concerned actions
        masks = np.zeros(
            (len(batch), self.action_count, self.qvalue_dim()),
            dtype=float)
        for i, (_, action, reward, _, terminal, _) in enumerate(batch):
            if self.has_scalar_qvalues():
                reward = np.dot(reward, w_batch[i])
            y_batch[i][action] = np.copy(reward)
            masks[i][action] = 1

            if not terminal:
                if self.alg == "naive":
                    for o in range(self.obj_cnt):
                        max_a = np.argmax(model_Qs[1::2][i][:, o])
                        y_batch[i][action][o] += self.discount * \
                            target_Qs[1::2][i][max_a][o]
                else:
                    if not self.has_scalar_qvalues():
                        max_a = np.argmax(
                            np.dot(model_Qs[1::2][i], w_batch[1::2][i]))
                    else:
                        max_a = np.argmax(model_Qs[1::2][i])
                    y_batch[i][action] += self.discount * \
                        target_Qs[1::2][i][max_a]

        inp = [w_batch[::2], x_batch, y_batch, masks]
        dummy = y_batch[:, 0, :]

        self.trainable_model.train_on_batch(inp, [dummy, y_batch])

        loss = self.update_priorities(batch, ids)
        self.recent_experiences = []

        return loss

    def train(self, environment, log_file, learning_steps, weights,
              per_weight_steps, total_steps):
        """Train agent on a series of weights for the given environment

        Arguments:
            environment {Object} -- Environment, should have reset() and step(action, frame_skip) functions
            log_file {string} -- Name of the log file
            learning_steps {int} -- Over how many steps epsilon is annealed
            weights {list} -- List of weight vectors
            per_weight_steps {int} -- How many steps each weight is active for, if 1, weights change at the end of each episode
            total_steps {int} -- Total training steps
        """

        self.epsilon = self.start_e
        per_weight_steps = per_weight_steps

        weight_index = 0
        self.steps = 0
        self.total_steps = total_steps

        # the amount of learning steps determines how quickly we start updating and how quickly epsilon is annealed
        self.learning_steps = max(1, learning_steps)

        self.env = environment
        self.log = Log(log_file, self)

        self.set_weights(weights[weight_index])

        model_pol_index = None
        episodes = 1
        episode_steps = 0
        pred_idx = None

        # current_state_raw = self.env.reset()
        # current_state = self.history.fill_raw_frame(
        #     current_state_raw["pixels"])
        
        # current_state_raw = self.env.reset()
        # current_state_raw_pixels = self.env.render(current_state_raw)
        # current_state = self.history.fill_raw_frame(current_state_raw_pixels)

        current_state_raw = self.env.reset()
        current_state = self.history.fill_raw_frame(current_state_raw)

        for i in range(int(self.total_steps)):

            self.steps = i
            episode_steps += 1

            # pick an action following an epsilon-greedy strategy
            action = self.pick_action(current_state)

            # perform the action
            # next_state_raw, reward, terminal = self.env.step(
            #     action, self.frame_skip)
            # next_state = self.history.add_raw_frame(next_state_raw["pixels"])

            # next_state_raw, reward, terminal, _ = self.env.step(action)
            # next_state_raw_pixels = self.env.render(next_state_raw)
            # next_state = self.history.add_raw_frame(next_state_raw_pixels)
            
            next_state_raw, reward, terminal, _ = self.env.step(action, self.frame_skip)
            next_state = self.history.add_raw_frame(next_state_raw)

            # memorize the experienced transition
            pred_idx = self.memorize(
                current_state,
                action,
                reward,
                next_state,
                terminal,
                self.steps,
                trace_id=episodes if self.memory_type == "DER" else self.steps,
                pred_idx=pred_idx)

            # update the networks and exploration rate
            loss = self.perform_updates(i)
            self.update_epsilon(i)

            if terminal or episode_steps > self.max_episode_length:
                # start_state_raw = self.env.reset()
                # next_state = self.history.fill_raw_frame(
                #     start_state_raw["pixels"])
                # start_state_raw = self.env.reset()
                # start_state_raw_pixels = self.env.render(start_state_raw)
                # next_state = self.history.fill_raw_frame(
                #     start_state_raw_pixels)
                start_state_raw = self.env.reset()
                next_state = self.history.fill_raw_frame(start_state_raw)

                pred_idx = None

            self.log.log_step(self.env, i, loss, reward,
                              terminal or episode_steps > self.max_episode_length, current_state, next_state,
                              self.weights, self.end_discount, episode_steps,
                              self.epsilon, self.frame_skip,
                              None, action)

            current_state = next_state
            current_state_raw = next_state_raw
            # current_state_raw_pixels = next_state_raw_pixels

            if terminal or episode_steps > self.max_episode_length:

                is_weight_change = int(
                    (i + 1) / per_weight_steps) != weight_index

                if self.alg == "mn" and is_weight_change:
                    # Compute the trained policy's value
                    cur_value = max(
                        self.predict(next_state),
                        key=lambda q: np.dot(q, self.weights))

                    # Add the trained policy
                    self.mn.add_new_policy(
                        cur_value, weights[:weight_index + 1], model_pol_index, override=weight_index<=1)

                if per_weight_steps == 1:
                    weight_index += 1
                else:
                    weight_index = int((i + 1) / per_weight_steps)

                if per_weight_steps == 1 or is_weight_change:
                    self.set_weights(weights[weight_index])

                # Load the past policy which is optimal for the new weight vector
                if self.alg == "mn" and is_weight_change:
                    model_pol_index = self.mn.load_optimal_policy(
                        self.weights, weight_index)

                episodes += 1
                episode_steps = 0
            
            if (self.steps % 200000) == 0:
                self.save_weights()

    def set_weights(self, weights):
        """Set current weight vector

        Arguments:
            weights {np.array} -- Weight vector of size N
        """

        self.weights = np.array(weights)
        self.weight_history.append(self.weights)

    def name(self):
        return self.extra

    def get_network_weights(self, start_layer=0, end_layer=-1):
        """Extracts weights of a subset of the model and target network's layer.
            Can for example be used to extract only the policy layer's parameters

        Keyword Arguments:
            start_layer {int} -- Start of the layers subset (default: {0})
            end_layer {int} -- End of the layers subset (default: {-1})

        Returns:
            tuple -- consisting of (model network subweights, target network subweights)
        """

        return ([
            layer.get_weights()
            for layer in self.model.layers[start_layer:end_layer]
        ], [
            layer.get_weights()
            for layer in self.target_model.layers[start_layer:end_layer]
        ])

    def initialize_memory(self):
        """Initialize the replay buffer, with a secondary diverse buffer and/or
            a secondary tree to store prediction errors
        """
        if self.memory_type == "DER" or self.memory_type == "SEL" or self.memory_type == "EXP":
            main_capacity = sec_capacity = self.memory_size // 2
        else:
            main_capacity, sec_capacity = self.memory_size, 0

        def der_trace_value(trace, trace_id, memory_indices):
            """Computes a trace's value as its return

            Arguments:
                trace {list} -- list of transitions
                trace_id {object} -- the trace's id
                memory_indices {list} -- list of the trace's indexes in memory

            Returns:
                np.array -- the trace's value
            """

            if trace_id in self.trace_values:
                return self.trace_values[trace_id]
            I_REWARD = 2
            value = np.copy(trace[0][I_REWARD])
            for i, v in enumerate(trace[1:]):
                value += v[I_REWARD] * self.end_discount**(i + 1)
            if type(value) == float:
                value = np.array([value])
            self.trace_values[trace_id] = value
            return value

        def sel_trace_value(trace, trace_id, memory_indices):
            assert len(trace) == 1
            if trace_id not in self.trace_values:
                self.trace_values[trace_id] = np.random.random()

            return self.trace_values[trace_id]

        def exp_trace_value(trace, trace_id, memory_indices):

            assert len(trace) == 1
            if trace_id not in self.trace_values:
                state, action, reward, next_state, terminal, extra = trace[0]
                q_values = self.predict(state)
                best_q = np.max(np.dot(q_values, self.weights))

                self.trace_values[trace_id] = best_q - \
                    np.dot(q_values[action], self.weights)

            return self.trace_values[trace_id]

        value_function = der_trace_value if self.memory_type == "DER" else sel_trace_value if self.memory_type == "SEL" else exp_trace_value
        trace_diversity = not(self.memory_type ==
                              "SEL" or self.memory_type == "EXP")
        self.buffer = DiverseMemory(
            main_capacity=main_capacity,
            sec_capacity=sec_capacity,
            value_function=value_function,
            trace_diversity=trace_diversity,
            a=self.mem_a,
            e=self.mem_e)

    def predict(self, state, model=None, weights=None):
        """Predict values for the given state

        Arguments:
            state {np.array} -- [The input state]

        Keyword Arguments:
            weights {np.array} -- [The weights input] (default: {active weights})
            model {Keras Model} -- [The model to predict with] (default: {self.model})

        Returns:
            np.array -- [Predicted values]
        """
        model = model or self.model
        weights = self.weights if weights is None else weights
        return model.predict([weights[np.newaxis, ], state[np.newaxis, ]])[0]

    def pick_action(self, state, weights=None):
        """Given a state and weights, compute the next action, following an
            epsilon-greedy strategy

        Arguments:
            state {np.array} -- The state in which to act

        Keyword Arguments:
            weights {np.array} -- The weights on which to act (default: self.weights)

        Returns:
            int -- The selected action's index
        """
        np.random.seed(self.steps)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)

        weights = self.weights if weights is None else weights
        self.q_values = self.predict(state, self.model, weights)
        if not self.has_scalar_qvalues():
            scalarized_qs = np.dot(self.q_values, weights)
        else:
            scalarized_qs = self.q_values
        return np.argmax(scalarized_qs)

    def perform_updates(self, steps):
        """Perform the necessary updates for the current number of steps, 
            The policy and the feature extraction (either implicitly or 
            explicitly depending on the architecture) are updated
        Arguments:
            steps {int} -- Number of steps

        Returns:
            loss -- The update loss
        """

        loss = 0
        if steps > self.min_buf_size * self.learning_steps:

            if steps % self.update_interval == 0:

                if self.is_first_update:
                    # Compute buffer's priorities all at once before the first
                    # update
                    self.recent_experiences = []
                    self.update_all_priorities()
                    self.is_first_update = False

                loss = self.policy_update()
            if steps % self.target_update_interval == 0:
                self.update_target()

        return loss
        
    def update_epsilon(self, steps):
        """Update exploration rate

        Arguments:
            steps {int} -- Elapsed number of steps
        """

        start_steps = self.learning_steps * self.start_annealing
        annealing_steps = self.learning_steps * \
            (1 - self.start_annealing)

        self.epsilon = linear_anneal(steps, annealing_steps, self.start_e,
                                     self.end_e, start_steps)
        self.discount = self.end_discount - linear_anneal(steps, annealing_steps, self.end_discount - self.start_discount,
                                                          0, start_steps)

    def memorize(self,
                 state,
                 action,
                 reward,
                 next_state,
                 terminal,
                 steps,
                 initial_error=0,
                 trace_id=None,
                 pred_idx=None):
        """Memorizes a transition into the replay, if no error is provided, the 
        transition is saved with the lowest priority possible, and should be
        updated accordingly later on.

        Arguments:
            state {Object} -- s_t
            action {int} -- a_t
            reward {np.array} -- r_t
            next_state {Object} -- s_{t+1}
            terminal {bool} -- wether s_{t+1} is terminal
            steps {int} -- t

        Keyword Arguments:
            initial_error {float} -- The initial error of the transition (default: {0})
            trace_id {object} -- The trace's identifier, if None, the transition is treated as
                                an individual trace. (default: {None})
        """
        if initial_error == 0 and not self.direct_update:
            initial_error = self.max_error
        extra = np.array(
            [steps, steps, self.epsilon, self.weights],
            dtype=object)

        transition = np.array((state, action, reward, next_state[-1], terminal,
                               extra))

        # Add transition to replay buffer

        idx = self.buffer.add(
            initial_error, transition, pred_idx=pred_idx, trace_id=trace_id)
        self.recent_experiences.append(idx)

        return idx

    def update_target(self):
        """Synchronize the target network
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_full_states(self, batch):
        """To save space, we only store the new part of the next state (i.e., 
        a single frame), this method reconstructs the next states from that 
        additional frame and from the previous state.

        Arguments:
            batch {list} -- Batch of transitions

        Returns:
            np.array -- Batch of reconstructed states, with every s_i stored in
                        position i*2 and s_{i+1} stored in position i*2+1
        """

        states = np.zeros(
            (len(batch) * 2,) + self.history.shape, dtype=np.uint8)
        for i, b in enumerate(batch):
            states[i * 2] = batch[i][0]
            states[i * 2 + 1][:-1] = batch[i][0][1:]
            states[i * 2 + 1][-1] = batch[i][3]
        return states

    def get_training_weights(self, batch):
        """Given a batch of transitions, this method generates a batch of
        weights to train on

        Arguments:
            batch {list} -- batch of transitions

        Returns:
            list -- batch of weights
        """

        w_batch = np.repeat([self.weights], len(batch), axis=0)
        if self.dupe in ("CN", "CN-UVFA"):
            if len(self.weight_history) > 1:
                max_index = len(
                    self.weight_history) - 1 if self.dupe in ("CN") else len(self.weight_history)

                idx = np.random.randint(
                    max_index, size=int(len(batch)))
                w_batch[::] = np.array(self.weight_history)[idx]

                if self.dupe in ("CN"):
                    w_batch[::2] = self.weights

        return w_batch

    def duplicate(self):
        return self.dupe in ("CN", "CN-UVFA", "CN-ACTIVE")

    def update_all_priorities(self):
        """Updates all priorities of the replay buffer
        """
        data = self.buffer.get_data(True)

        chunk_size = 100
        for i in range(0, len(data[0]), chunk_size):
            chunk_data = np.array(data[1][i:i + chunk_size])
            chunk_ids = data[0][i:i + chunk_size]

            if self.duplicate():
                chunk_data = np.repeat(chunk_data, 2, axis=0)
                chunk_ids = np.repeat(chunk_ids, 2, axis=0)

    def get_training_data(self, batch):
        """Given a batch of transitions, this method reconstructs the states,
        generates a batch of weights and computes the model and target predictions for both each s_i and s_{i+1}

        Arguments:
            batch {list} -- batch of transitions

        Returns:
            tuple -- (model predictions, target predictions, weight batch, states)
        """
        states = self.get_full_states(batch)

        w_batch = self.get_training_weights(batch)
        w_batch = np.repeat(w_batch, 2, axis=0)
        inp = [w_batch, states]

        # Predict both the model and target q-values
        model_q = self.model.predict(inp, batch_size=200)
        target_q = self.target_model.predict(inp, batch_size=200)

        return model_q, target_q, w_batch, states

    def update_priorities(self, batch, ids, ignore_dupe=False, pr=False):
        """Given a batch of transitions, this method computes each transition's
        error and uses that error to update its priority in the replay buffer

        Arguments:
            batch {list} -- list of transitions
            ids {list} -- list of identifiers of each transition in the replay
                        buffer

        Returns:
            float -- The batch's mean loss
        """

        model_q, target_q, weights, _ = self.get_training_data(batch)

        errors = np.zeros(len(batch))

        for i, (_, action, reward, _, terminal,
                extra) in enumerate(batch):

            target = np.copy(reward)

            if self.has_scalar_qvalues():
                target = np.dot(target, weights[i * 2])

            if not terminal:
                if self.alg == "naive":
                    for o in range(self.obj_cnt):
                        max_a = np.argmax(model_q[i * 2 + 1][:, o])
                        target[o] += self.discount * \
                            target_q[i * 2 + 1][max_a][o]
                else:
                    if self.has_scalar_qvalues():
                        next_action = np.argmax(model_q[i * 2 + 1])
                    else:
                        next_action = np.argmax(
                            np.dot(model_q[i * 2 + 1], weights[i * 2 + 1]))
                    target += self.discount * \
                        target_q[i * 2 + 1][next_action]
            print(f"predictive:{model_q[i * 2][action]}")
            print(f"target:{target}")
            error = mae(model_q[i * 2][action], target)

            errors[i] = error

            # When dupe is True, we train each sample on two weight vectors
            # Hence, there are two TD-errors per sample, we use the mean of
            # both errors to update the priorities
            if self.duplicate():
                if i % 2 == 0:
                    continue
                error = (error + errors[i - 1]) / 2
            self.buffer.update(ids[i], error)
            self.max_error = max(error, self.max_error)

        return np.mean(errors)

    def has_scalar_qvalues(self):
        return "scal" in self.alg or "uvfa" in self.alg

    def save_weights(self):
        """Saves the networks' weights to files identified by the agent's name
        and the current weight vector
        """

        self.model.save_weights(
            "output/networks/{}_{}.weights".format(self.name(), self.weights))



if __name__ == "__main__":

    # dst = gym.make('BountifulSeaTreasure-v1')
    dst = DeepSeaTreasure(view=(5,5), full=True, scale=1)
    obj_cnt = 2
    steps = 10000
    all_weights = generate_weights(count=steps, n=obj_cnt, m=1)
    # shape (100000, 2)
    # print(np.array(all_weights).shape)

    parser = OptionParser()
    parser.add_option(
        "-l",
        "--algorithm",
        dest="alg",
        choices=["scal", "mo", "mn", "cond", "uvfa", "random", "naive"],
        default="cond",
        help="Architecture type, one of 'scal','mo','meta','cond'")
    parser.add_option(
        "-m",
        "--memory",
        dest="mem",
        default="DER",
        choices=["STD", "DER", "SEL", "EXP"],
        help="Memory type, one of 'std','crowd','exp','sel'")
    parser.add_option(
        "-d",
        "--dupe",
        dest="dupe",
        default="CN",
        choices=["none", "CN", "CN-UVFA", "CN-ACTIVE"],
        help="Extra training")
    parser.add_option(
        "--end_e",
        dest="end_e",
        default="0.01",
        help="Final epsilon value",
        type=float)
    parser.add_option(
        "--start_e",
        dest="start_e",
        default="0.1",
        help="start epsilon value",
        type=float)
    parser.add_option(
        "-r", "--lr", dest="lr", default="0.02", help="learning rate", type=float)
    parser.add_option(
        "--clipnorm", dest="clipnorm", default="1", help="clipnorm", type=float)
    parser.add_option(
        "--mem-a", dest="mem_a", default="2.", help="memory error exponent", type=float)
    parser.add_option(
        "--mem-e", dest="mem_e", default="0.01", help="error offset", type=float)
    parser.add_option(
        "--clipvalue",
        dest="clipvalue",
        default="0.5",
        help="clipvalue",
        type=float)
    parser.add_option(
        "--momentum", dest="momentum", default="0.9", help="momentum", type=float)
    parser.add_option(
        "-u",
        "--update_period",
        dest="updates",
        default="4",
        help="Update interval",
        type=int)
    parser.add_option(
        "--target-update",
        dest="target_update_interval",
        default="150",
        help="Target update interval",
        type=int)
    parser.add_option(
        "-f",
        "--frame-skip",
        dest="frame_skip",
        default="1",
        help="Frame skip",
        type=int)
    parser.add_option(
        "--sample-size",
        dest="sample_size",
        default="16",
        help="Sample batch size",
        type=int)
    parser.add_option(
        "-g",
        "--discount",
        dest="discount",
        default="0.95",
        help="Discount factor",
        type=float)
    parser.add_option("--scale", dest="scale", default=1,
                    help="Scaling", type=float)
    parser.add_option("--anneal-steps",
                    dest="steps", default=10000, help="steps",  type=int)
    parser.add_option("-x", "--extra", dest="extra", default="")
    parser.add_option("-p", "--reuse", dest="reuse",
                    choices=["full", "sectionned", "proportional"], default="full")
    parser.add_option(
        "-c", "--mode", dest="mode", choices=["regular", "sparse"], default="sparse")
    parser.add_option(
        "-s", "--seed", dest="seed", default=None, help="Random Seed", type=int)
    parser.add_option(
        "--lstm", dest="lstm", default=False)

    (options, args) = parser.parse_args()

    
    extra = "test-newEnv-lstm-{} clipN-{} clipV-{} a-{} m-{} s-{}  e-{} d-{} x-{} {} p-{} fs-{} d-{} up-{} lr-{} e-{} p-{} m-{}-{}".format(
    options.lstm, options.clipnorm, options.clipvalue,
    options.alg, options.mem, options.seed,
    options.end_e, options.dupe, options.extra, options.mode, options.reuse,
    options.frame_skip,
    np.round(options.discount, 4), options.updates,
    np.round(options.lr, 4),
    np.round(options.scale, 2), np.round(options.steps, 2), np.round(options.mem_a, 2), np.round(options.mem_e, 2))

    agent = DeepAgent(
        range(4), #range(ACTION_COUNT). e.g. range(6)
        obj_cnt,
        # steps, #memory size
        # sample_size=16,
        # weights=None,
        # discount=0.95,
        # learning_rate=0.02,
        # target_update_interval=150,
        # alg="scal",
        # frame_skip=1,
        # start_e=0.1,
        # end_e=0.01,
        # memory_type="DER",
        # update_interval=4,
        # reuse="full",
        # mem_a=2.,
        # mem_e=0.01,
        # extra=extra,
        # clipnorm=0,
        # clipvalue=0,
        # momentum=0.9,
        # scale=1,
        # dupe="CN"
        
        options.steps,
        sample_size=options.sample_size,
        weights=None,
        discount=options.discount,
        learning_rate=options.lr,
        target_update_interval=options.target_update_interval,
        alg=options.alg,
        frame_skip=options.frame_skip,
        start_e=options.start_e,
        end_e=options.end_e,
        memory_type=options.mem,
        update_interval=options.updates,
        reuse=options.reuse,
        mem_a=options.mem_a,
        mem_e=options.mem_e,
        extra=extra,
        clipnorm=options.clipnorm,
        clipvalue=options.clipvalue,
        momentum=options.momentum,
        scale=options.scale,
        dupe=None if options.dupe == "none" else options.dupe,
        lstm=options.lstm)

    steps_per_weight = 5000 if options.mode == "sparse" else 1
    log_file = open('output/logs/rewards_{}'.format(extra), 'w', 1)
    agent.train(dst, log_file, 100000, all_weights, steps_per_weight, options.steps*10)