"""Classes of networks

Author: Xiaodong Nian
Time: November 2019
"""

import numpy as np
from numpy.core.fromnumeric import trace
import tensorflow as tf
from collections import deque

from networks import Actor, Critic, Critic0, ActorCritic, ActorCriticER
# NN configs.
from config_agent import IM_SIZE, BLACK_AND_WHITE
# from diverse_mem import DiverseMemory
from diverse_mem_attentive import AttentiveMemoryBuffer, MemoryBuffer
from history import History
from utils import *

class DCRACAgent:
    def __init__(self,
                 env,
                #  pixel_env, pixel_env is for minecart env
                 gamma=0.98,
                 weights=None,
                 timesteps=5,
                 batch_size=32,
                 replay_type='STD',
                 buffer_size=10000,
                 buffer_a=2,
                 buffer_e=0.01,
                 memnn_size=9,
                 start_annealing=0.05,
                 max_episode_length=500,
                 start_e=1.,
                 end_e=0.05,
                 im_size=(IM_SIZE, IM_SIZE),
                 grayscale=BLACK_AND_WHITE,
                 net_type='R',
                 obj_func='a',
                 lr=1e-3,
                 lr_2=1e-3,
                 frame_skip=4,
                 update_interval=4,
                 clipnorm=1,
                 clipvalue=1,
                 nesterov=True,
                 momentum=0.9,
                 dup=False,
                 min_buf_size=0.1,
                 action_conc=True,
                 feature_embd=True,
                 extra=None,
                 gpu_setting='1'):
        
        self.env = env
        # self.pixel_env = pixel_env
        # those are for dst env
        self.nb_action = self.env.action_space.shape[0]
        self.observation_shape = self.env.observation_space.shape
        self.nb_objective = self.env.obj_cnt

        # those are for minecart
        # self.nb_action = len(self.env.action_space())
        # self.observation_shape = self.env.observation_space.shape
        # self.nb_objective = self.env.obj_cnt()

        self.discount = gamma
        self.qvalue_dim = self.nb_objective

        self.timesteps = timesteps
        self.batch_size = batch_size
        self.replay_type = replay_type
        self.buffer_size = buffer_size
        self.buffer_a = buffer_a
        self.buffer_e = buffer_e
        self.max_episode_length = max_episode_length

        self.start_lambda = 3
        self.end_lambda = 1
        self.alpha = 1
        
        self.fill_history = False
        self.stoch_policy = True

        self.memnn_size = memnn_size

        self.weight_history = []
        if weights is not None:
            self.set_weights(np.array(weights))
        
        self.start_e = start_e
        self.end_e = end_e
        self.start_annealing = start_annealing

        self.im_size = im_size
        channel = 1 if grayscale else 3
        self.im_shape = im_size + (channel, )

        self.lr_2 = lr_2
        self.lr = lr
        self.frame_skip = frame_skip
        
        self.net_type = net_type
        self.action_conc = action_conc
        self.feature_embd = feature_embd

        # self.mask_empty = True if self.action_conc and self.net_type in ('R', 'M') else False
        self.mask_empty = True

        self.dup = dup
        self.obj_func = obj_func.lower()
        self.min_buf_size = min_buf_size
        self.update_interval = update_interval
        self.actor_update_interval = 1

        self.nesterov = nesterov
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.momentum = momentum

        self.direct_update = True
        self.is_first_update = True
        self.max_error = .0
        self.trace_values={}
        self.recent_experiences = []

        self.extra = extra
        self.gpu_setting = gpu_setting

        # Tensorflow GPU optimization.
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        # from keras import backend as K
        tf.compat.v1.keras.backend.set_session(sess)

        # Initialize the history 
        self.history = History(self.timesteps, self.im_shape, self.nb_action)

        # Make neural networks
        self.build_models()

        # Initialize replay buffer.
        self.initialize_memory()


    def build_models(self):
        # Make actor network (creates target model internally).
        self.actor = Actor(self.nb_action, self.im_shape, self.nb_objective, self.qvalue_dim,
            timesteps=self.timesteps, net_type=self.net_type, memnn_size=self.memnn_size, 
            lr=self.lr_2, action_conc=self.action_conc, action_embd=self.feature_embd, 
            weight_embd=self.feature_embd, backendtype=self.gpu_setting)

        # Make critic network (creates target model internally).
        self.critic = Critic(self.nb_action, self.im_shape, self.nb_objective, self.qvalue_dim,
            timesteps=self.timesteps, net_type=self.net_type, memnn_size=self.memnn_size,
            lr=self.lr, action_conc=self.action_conc, action_embd=self.feature_embd, 
            weight_embd=self.feature_embd, backendtype=self.gpu_setting)
        
        # log the settings
        with open("output/logs/info_"+self.extra+".log", "a") as f:
            f.write(time.asctime()+'\n\n')
            f.write('ACTORRRRRRRRRRRRRRRRRRRRRRR\n')
            f.write(str(self.actor.model.get_config())+'\n\n')
            f.write('CRITICCCCCCCCCCCCCCCCCCCCCC\n')
            f.write(str(self.critic.model.get_config())+'\n\n')
            f.write(str(vars(self))+'\n\n')
        print(vars(self))


    # Train or run for some number of episodes.
    def train(self, log_file, learning_steps, weights, 
              per_weight_steps, total_steps, log_game_step=False):
        
        self.learning_steps = learning_steps
        self.epsilon = self.start_e
        per_weight_steps = per_weight_steps

        weight_index = 0
        self.steps = 0
        self.total_steps = total_steps

        self.log = Log(log_file)
        self.log_game_step = log_game_step

        self.set_weights(weights[weight_index])

        episodes = 1
        episode_steps = 0
        pred_idx = None

        current_state_raw = self.env.reset()
        self.current_state, last_action = self.history.reset_with_raw_frame(current_state_raw, fill=self.fill_history)

        for i in range(int(self.total_steps)):

            self.steps = i
            episode_steps += 1

            # pick an action following an epsilon-greedy strategy
            action, acts_prob = self.pick_action(self.current_state, last_action)

            # perform the action
            next_state_raw, reward, terminal, info = self.env.step(action, self.frame_skip)
            next_state, next_last_action = self.history.add_raw_frame(next_state_raw, action)

            if self.log_game_step:
                print("Taking action", action, "under prob", acts_prob, "at", info["position"], "with reward", reward)

            # memorize the experienced transition
            pred_idx = self.memorize(
                self.current_state,
                action,
                reward,
                next_state,
                terminal,
                last_action,
                acts_prob,
                trace_id=episodes if self.replay_type == "DER" else self.steps,
                pred_idx=pred_idx)

            # update the networks and exploration rate
            self.update_lambda(i)
            loss = self.perform_updates(i)
            self.update_epsilon(i)

            self.log.log_step(episodes, i, loss, reward,
                              terminal or episode_steps > self.max_episode_length, 
                              self.weights, self.discount, episode_steps,
                              self.epsilon, self.frame_skip, action)

            self.current_state = next_state
            current_state_raw = next_state_raw
            last_action = next_last_action
            
            if terminal or episode_steps > self.max_episode_length:
                current_state_raw = self.env.reset()
                self.current_state, last_action = self.history.reset_with_raw_frame(current_state_raw, fill=self.fill_history)
                pred_idx = None

                is_weight_change = int(
                    (i + 1) / per_weight_steps) != weight_index

                if per_weight_steps == 1:
                    weight_index += 1
                else:
                    weight_index = int((i + 1) / per_weight_steps)

                if per_weight_steps == 1 or is_weight_change:
                    self.set_weights(weights[weight_index])

                episodes += 1
                episode_steps = 0

            if (i + 1) % 10000 == 0:
                # self.save_weights()
                self.save_model()


    def set_weights(self, weights):
        """Set current weight vector

        Arguments:
            weights {np.array} -- Weight vector of size N
        """

        self.weights = np.array(weights)
        self.weight_history.append(self.weights)


    def pick_action(self, trace_o, trace_a, weights=None):
        """Given a observation trace and weights, compute the next action, following an
            epsilon-greedy strategy

        Arguments:
            trace_o {np.array} -- The trace of observations o_t in which to act 
            trace_a {np.array} -- The trace of actions a_{t-1}, in the same order as trace_o

        Keyword Arguments:
            weights {np.array} -- The weights on which to act (default: self.weights)

        Returns:
            int -- The selected action's index
        """
        # np.random.seed(self.steps)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.action_space), (np.ones(self.nb_action)/self.nb_action)

        trace_a = self._get_onehot_action_trace(trace_a)
        weights = self.weights if weights is None else weights

        action, acts_prob = self.actor.choose_action(
            weights[np.newaxis, ], trace_o[np.newaxis, ], trace_a[np.newaxis, ], 
            stochastic=self.stoch_policy, return_probs=True)
        
        return action, acts_prob


    def _get_onehot_action_trace(self, actions):
        """Translate the numerical actions array as one-hot array.
        
        Args:
            actions ([int]): integer action list

        Returns:
            np.array: one-hot action list
        """
        length = len(actions)
        onehot_trace = np.zeros((length, self.nb_action))
        for i, a in enumerate(actions):
            if a >= 0:
                onehot_trace[i][a] = 1
            # if mask_empty, set all action slots as -1 to be masked by the network.
            if self.mask_empty and a == -1:
                onehot_trace[i] = -1
        return onehot_trace


    # Given a bunch of experienced histories, update our models.
    def perform_updates(self, steps):
        loss = 0
        if steps> self.min_buf_size * self.learning_steps:

            if steps % self.update_interval == 0:

                if self.is_first_update:
                    # Compute buffer's priorities all at once before the first update
                    self.recent_experiences = []
                    self.update_all_priorities()
                    self.is_first_update = False

                loss = self.policy_update(update_actor=(steps%(self.actor_update_interval*self.update_interval) == 0))
                self.update_target(soft_update=True)

        return loss

    def update_target(self, **kwargs):
        self.actor.update_target(**kwargs)
        self.critic.update_target(**kwargs)

    def policy_update(self, update_actor=True):
        # np.random.seed(self.steps)

        ids, batch, _ = self.buffer.sample(self.batch_size)
        # ids, batch, _ = self.buffer.sampel(self.sample_size, self.k, self.steps, self.wegihts, self.current_state)

        if self.direct_update:
            # Add recent experiences to the priority update batch
            batch = np.concatenate((batch, self.buffer.get(self.recent_experiences)), axis=0)
            ids = np.concatenate((ids, self.recent_experiences)).astype(int)

        # if self.dup is True, we train each sample in the batch on two
        # weight vectors, hence we duplicate the batch data
        if self.dup:
            batch = np.repeat(batch, 2, axis=0)
            ids = np.repeat(ids, 2, axis=0)

        cur_batch_size = len(batch)
        
        # q_predict, q_target, p_predict, p_target, w_batch, o_batch, a_last_batch
        q_predict, q_target, p_predict, p_target, w_batch, o_batch, a_last_batch = self._get_training_data(batch)

        q_true = np.copy(q_predict)
        q_mask = np.zeros((cur_batch_size, self.nb_action, self.qvalue_dim), dtype=float)
        action_objective = np.zeros((cur_batch_size, self.nb_action), dtype=float)
        
        for i, (_, action, reward, _, done, _, _) in enumerate(batch):
            q_true[i][action] = np.copy(reward)
            q_mask[i][action] = 1

            if not done:
                if self.stoch_policy:
                    # # stochastic policy (sample an action 'a_next' with target action_probs)
                    # a_next = np.random.choice(range(self.nb_action), p=p_target[i])
                    # q_true[i][action] += self.discount * q_target[i][a_next]

                    # Expectation for stochastic policy
                    q_true[i][action] += self.discount * np.dot(p_target[i], q_target[i])
                else:
                    # deterministic policy (choose the 'a_next' with largest target action_probs)
                    a_next = np.argmax(p_target[i])
                    q_true[i][action] += self.discount * q_target[i][a_next]

            if update_actor:
                action_objective[i][action] = self._get_action_objective(q_true[i], q_predict[i], p_predict[i], action, w_batch[i])

        a_batch = self._get_onehot_action_trace([b[1] for b in batch])
        p_old_batch = np.array([b[6] for b in batch])
            
        self.train_on_batch(w_batch, o_batch, a_last_batch, q_true, action_objective, a_batch, p_old_batch, q_mask)

        loss = self.update_priorities(batch, ids)
        self.recent_experiences = []
    
        return loss

    def train_on_batch(self, w_batch, o_batch, a_last_batch, q_true, action_objective, a_batch, p_old_batch, q_mask=None):
        self.critic.model.train_on_batch([w_batch, o_batch, a_last_batch], q_true)
        self.actor.model.train_on_batch([w_batch, o_batch, a_last_batch], action_objective)

    def _get_training_data(self, batch):

        o_batch, o_next_batch, a_last_batch, a_last_next_batch = self._get_full_states(batch)

        w_batch = self.get_training_weights(batch)

        # Predict both the model and target q-values
        predict_input = [w_batch, o_batch, a_last_batch]
        target_input = [w_batch, o_next_batch, a_last_next_batch]

        # Predict both the model and target q-values
        q_predict = self.critic.model.predict(predict_input)
        q_target = self.critic.target_model.predict(target_input)

        # Predict both the model and target action probs
        p_predict = self.actor.model.predict(predict_input)
        p_target = self.actor.target_model.predict(target_input)

        return q_predict, q_target, p_predict, p_target, w_batch, o_batch, a_last_batch
    
    def _get_full_states(self, batch):

        o_batch = np.zeros((len(batch),) + self.history.history_o.shape, dtype=np.uint8)
        o_next_batch = np.zeros((len(batch),) + self.history.history_o.shape, dtype=np.uint8)

        # one-hot for predict
        a_last_batch = np.zeros((len(batch), self.history.length, self.nb_action), dtype=float)
        a_last_next_batch = np.zeros((len(batch), self.history.length, self.nb_action), dtype=float)

        for i, b in enumerate(batch):
            o_batch[i] = b[0]
            o_next_batch[i][:-1] = b[0][1:] 
            o_next_batch[i][-1] = b[3]
            
            a_last_batch[i] = self._get_onehot_action_trace(b[5])
            a_last_next_batch[i] = np.roll(a_last_batch[i], -1, axis=0)
            a_last_next_batch[i][-1][b[1]] = 1

        return o_batch, o_next_batch, a_last_batch, a_last_next_batch
    
    def _get_action_objective(self, q_true, q_predict, p_predict, action, weight):
        if self.obj_func == 'a':
            # based on advantages
            objective = q_true[action] - np.average(q_predict, weights=p_predict, axis=0)
        elif self.obj_func == 'am':
            objective = q_true[action] - q_predict.mean(axis=0)
        elif self.obj_func == 'td':
            # based on td error
            objective = q_true[action] - q_predict[action]
        elif self.obj_func == 'y':
            # based on y values
            objective = q_true[action]
        else:
            # based on q values
            objective = q_predict[action]

        return np.dot(objective, weight)
    
    def get_training_weights(self, batch):
        """Given a batch of transitions, this method generates a batch of
        weights to train on

        Arguments:
            batch {list} -- batch of transitions

        Returns:
            list -- batch of weights
        """
        w_batch = np.repeat([self.weights], len(batch), axis=0)
        if self.dup and len(self.weight_history) > 1:
            idx = np.random.randint(len(self.weight_history)-1 , size=int(len(batch)))
            w_batch[::] = np.array(self.weight_history)[idx]
            w_batch[::2] = self.weights
        return w_batch

    def update_priorities(self, batch, ids, ignore_dup=False, pr=False):
        """Given a batch of transitions, this method computes each transition's
        error and uses that error to update its priority in the replay buffer

        Arguments:
            batch {list} -- list of transitions
            ids {list} -- list of identifiers of each transition in the replay
                        buffer

        Returns:
            float -- The batch's mean loss
        """ 
        # q_predict, q_target, p_predict, p_target, w_batch, o_batch, a_last_batch
        q_predict, q_target, _, p_target, _, _, _ = self._get_training_data(batch)

        errors = np.zeros(len(batch))

        for i, (_, action, reward, _, terminal, _, _) in enumerate(batch):
            
            target = np.copy(reward)

            if not terminal:
                if self.stoch_policy:
                    # # stochastic policy (sample an action 'a_next' with target action_probs)
                    # a_next = np.random.choice(range(self.nb_action), p=p_target[i])
                    # target += self.discount * q_target[i][a_next]

                    # Expectation for stochastic policy
                    target += self.discount * np.dot(p_target[i], q_target[i])
                else:
                    # deterministic policy (choose the 'a_next' with largest target action_probs)
                    a_next = np.argmax(p_target[i])
                    target += self.discount * q_target[i][a_next]
                
            error = mae(q_predict[i][action], target)
            errors[i] = error

            # When dup is True, we train each sample on two weight vectors
            # Hence, there are two TD-errors per sample, we use the mean of
            # both errors to update the priorities
            if self.dup:
                if i % 2 == 0:
                    continue
                error = (error + errors[i - 1]) / 2
            self.buffer.update(ids[i], error)
            self.max_error = max(error, self.max_error)

        return np.mean(errors)

    def update_all_priorities(self):
        """Updates all priorities of the replay buffer
        """
        data = self.buffer.get_data(True)

        chunk_size = 100
        for i in range(0, len(data[0]), chunk_size):
            chunk_data = np.array(data[1][i:i + chunk_size])
            chunk_ids = data[0][i:i + chunk_size]

            if self.dup:
                chunk_data = np.repeat(chunk_data, 2, axis=0)
                chunk_ids = np.repeat(chunk_ids, 2, axis=0)
    
    def update_epsilon(self, steps):
        """Update exploration rate

        Arguments:
            steps {int} -- Elapsed number of steps
        """

        start_steps = self.learning_steps * self.start_annealing
        annealing_steps = self.learning_steps * (1 - self.start_annealing)

        self.epsilon = linear_anneal(steps, annealing_steps, self.start_e, self.end_e, start_steps)

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

    def update_lambda(self, steps):
        start_steps = self.learning_steps * self.start_annealing
        annealing_steps = self.total_steps * self.alpha

        self.k = self.linear_anneal_lambda(steps, annealing_steps, self.start_lambda, self.end_lambda, start_steps)

    def linear_anneal_lambda(self, steps, annealing_steps, start_lambda, end_lambda, start_steps):
        t = max(0, steps - start_steps)
        return max(end_lambda, (annealing_steps-t) * (start_lambda - end_lambda) / annealing_steps + end_lambda)


    def initialize_memory(self):
        """Initialize the replay buffer, with a secondary diverse buffer and/or
            a secondary tree to store prediction errors
        """
        if self.replay_type == "DER":
            main_capacity = sec_capacity = self.buffer_size // 2
        else:
            main_capacity, sec_capacity = self.buffer_size, 0

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
                value += v[I_REWARD] * self.discount**(i + 1)
            if type(value) == float:
                value = np.array([value])
            self.trace_values[trace_id] = value
            return value

        self.buffer = MemoryBuffer(main_capacity=main_capacity, sec_capacity=sec_capacity,
            value_function=der_trace_value, trace_diversity=True, a=self.buffer_a, e=self.buffer_e)

        # self.buffer = AttentiveMemoryBuffer(main_capacity=main_capacity, sec_capacity=sec_capacity,
        #     value_function=der_trace_value, trace_diversity=True, a=self.mem_a, e=self.mem_e)

    def memorize(self, state, action, reward, next_state, terminal, action_prev, acts_prob, 
        initial_error=0, trace_id=None, pred_idx=None):
        """Memorizes a transition into the replay, if no error is provided, the 
        transition is saved with the lowest priority possible, and should be
        updated accordingly later on.

        Arguments:
            state {object} -- s_t
            action {int} -- a_t
            reward {np.array} -- r_t
            next_state {object} -- s_{t+1}
            terminal {bool} -- wether s_{t+1} is terminal
            action_prev {object} -- a_{t-1}
            acts_prob {np.array} -- sample probability

        Keyword Arguments:
            initial_error {float} -- The initial error of the transition (default: {0})
            trace_id {object} -- The trace's identifier, if None, the transition is treated as
                                an individual trace. (default: {None})
        """
        if initial_error == 0 and not self.direct_update:
            initial_error = self.max_error

        transition = np.array((state, action, reward, next_state[-1], terminal, action_prev, acts_prob))

        # Add transition to replay buffer
        idx = self.buffer.add(initial_error, transition, pred_idx=pred_idx, trace_id=trace_id)
        self.recent_experiences.append(idx)

        return idx
        
    @property
    def name(self):
        return self.extra

    def save_weights(self):
        """Saves the networks' weights to files identified by the agent's name
        and the current weight vector
        """
        self.critic.model.save_weights("output/networks/{}_critic.weights".format(self.name))
        self.actor.model.save_weights("output/networks/{}_actor.weights".format(self.name))

    def save_model(self):
        """Saves the networks to files identified by the agent's name
        and the current weight vector
        """
        self.critic.model.save("output/networks/{}_critic_model.h5".format(self.name))
        self.actor.model.save("output/networks/{}_actor_model.h5".format(self.name))






class DCRACSAgent(DCRACAgent):
    def build_models(self):
        # Make only critic network (creates target model internally).
        self.actorcritic = ActorCritic(self.nb_action, self.im_shape, self.nb_objective, self.qvalue_dim,
            timesteps=self.timesteps, net_type=self.net_type, memnn_size=self.memnn_size, 
            lr=self.lr, action_conc=self.action_conc, action_embd=self.feature_embd, 
            weight_embd=self.feature_embd, actor_lr=self.lr_2, backendtype=self.gpu_setting)

        with open("output/logs/info_"+self.extra+".log", "a") as f:
            f.write(time.asctime()+'\n\n')
            f.write(str(self.actorcritic.model.get_config())+'\n\n')
            f.write(str(vars(self))+'\n\n')
        print(vars(self))

    def pick_action(self, trace_o, trace_a, weights=None):
        # np.random.seed(self.steps)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.action_space), (np.ones(self.nb_action)/self.nb_action)

        trace_a = self._get_onehot_action_trace(trace_a)
        weights = self.weights if weights is None else weights

        action, acts_prob = self.actorcritic.choose_action(
            weights[np.newaxis, ], trace_o[np.newaxis, ], trace_a[np.newaxis, ], 
            stochastic=self.stoch_policy, return_probs=True)

        return action, acts_prob

    def update_target(self, **kwargs):
        self.actorcritic.update_target(**kwargs)

    def train_on_batch(self, w_batch, o_batch, a_last_batch, q_true, action_objective, a_batch, p_old_batch, q_mask=None):
        self.actorcritic.model.train_on_batch([w_batch, o_batch, a_last_batch], [q_true, action_objective])

    def _get_training_data(self, batch):

        o_batch, o_next_batch, a_last_batch, a_last_next_batch = self._get_full_states(batch)

        w_batch = self.get_training_weights(batch)

        # Predict both the model and target q-values
        predict_input = [w_batch, o_batch, a_last_batch]
        target_input = [w_batch, o_next_batch, a_last_next_batch]

        # Predict both the model and target q-values
        q_predict, p_predict = self.actorcritic.model.predict(predict_input)
        q_target, p_target = self.actorcritic.target_model.predict(target_input)

        return q_predict, q_target, p_predict, p_target, w_batch, o_batch, a_last_batch

    def save_weights(self):
        """Saves the networks' weights to files identified by the agent's name
        and the current weight vector
        """
        self.actorcritic.model.save_weights("output/networks/{}_actorcritic_share.weights".format(self.name))

    def save_model(self):
        """Saves the networks to files identified by the agent's name
        and the current weight vector
        """
        self.actorcritic.model.save("output/networks/{}_critic_model.h5".format(self.name))



class DCRACSEAgent(DCRACSAgent):
    def build_models(self):
        # Make only critic network (creates target model internally).
        self.actorcritic = ActorCriticER(self.nb_action, self.im_shape, self.nb_objective, self.qvalue_dim,
            timesteps=self.timesteps, net_type=self.net_type, memnn_size=self.memnn_size, 
            lr=self.lr, action_conc=self.action_conc, action_embd=self.feature_embd, 
            weight_embd=self.feature_embd, backendtype=self.gpu_setting)

        with open("output/logs/info_"+self.extra+".log", "a") as f:
            f.write(time.asctime()+'\n\n')
            f.write(str(self.actorcritic.model.get_config())+'\n\n')
            f.write(str(vars(self))+'\n\n')
        print(vars(self))

    def train_on_batch(self, w_batch, o_batch, a_last_batch, q_true, action_objective, a_batch, p_old_batch, q_mask=None):
        action_objective = np.sum(action_objective, axis=1)
        self.actorcritic.trainable_model.train_on_batch([w_batch, o_batch, a_last_batch, action_objective, p_old_batch], [q_true, a_batch])



class DCRAC0Agent(DCRACAgent):
    def build_models(self):
        # Make actor network (creates target model internally).
        self.actor = Actor(self.nb_action, self.im_shape, self.nb_objective, self.qvalue_dim,
            timesteps=self.timesteps, net_type=self.net_type, memnn_size=self.memnn_size, 
            lr=self.lr_2, action_conc=self.action_conc, action_embd=self.feature_embd, 
            weight_embd=self.feature_embd, backendtype=self.gpu_setting)

        # Make critic network (creates target model internally).
        self.critic = Critic0(self.nb_action, self.im_shape, self.nb_objective, self.qvalue_dim,
            timesteps=self.timesteps, net_type=self.net_type, memnn_size=self.memnn_size, 
            lr=self.lr, action_conc=self.action_conc, action_embd=self.feature_embd, 
            weight_embd=self.feature_embd, backendtype=self.gpu_setting)

        # log the settings
        with open("output/logs/info_"+self.extra+".log", "a") as f:
            f.write(time.asctime()+'\n\n')
            f.write('ACTORRRRRRRRRRRRRRRRRRRRRRR\n')
            f.write(str(self.actor.model.get_config())+'\n\n')
            f.write('CRITICCCCCCCCCCCCCCCCCCCCCC\n')
            f.write(str(self.critic.model.get_config())+'\n\n')
            f.write(str(vars(self))+'\n\n')
        print(vars(self))

    def train_on_batch(self, w_batch, o_batch, a_last_batch, q_true, action_objective, a_batch, p_old_batch, q_mask=None):
        dummy = q_true[:, 0, :]
        self.critic.trainable_model.train_on_batch([w_batch, o_batch, a_last_batch, q_true, q_mask], [dummy, q_true])
        self.actor.model.train_on_batch([w_batch, o_batch, a_last_batch], action_objective)


class CNAgent(DCRACAgent):
    def build_models(self):
        # Make only critic network (creates target model internally).
        self.critic = Critic(self.nb_action, self.im_shape, self.nb_objective, self.qvalue_dim,
            timesteps=self.timesteps, net_type=self.net_type, memnn_size=self.memnn_size,
            lr=self.lr, action_conc=self.action_conc, action_embd=self.feature_embd, 
            weight_embd=self.feature_embd, backendtype=self.gpu_setting)

        with open("output/logs/info_"+self.extra+".log", "a") as f:
            f.write(time.asctime()+'\n\n')
            f.write(str(self.critic.model.get_config())+'\n\n')
            f.write(str(vars(self))+'\n\n')
        print(vars(self))
    
    def pick_action(self, trace_o, trace_a, weights=None):
        """Given a observation trace and weights, compute the next action, following an
            epsilon-greedy strategy

        Arguments:
            trace_o {np.array} -- The trace of observations o_t in which to act 
            trace_a {np.array} -- The trace of actions a_{t-1}, in the same order as trace_o

        Keyword Arguments:
            weights {np.array} -- The weights on which to act (default: self.weights)

        Returns:
            int -- The selected action's index
        """
        # np.random.seed(self.steps)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.action_space), (np.ones(self.nb_action)/self.nb_action)

        trace_a = self._get_onehot_action_trace(trace_a)
        weights = self.weights if weights is None else weights

        q_value = self.critic.model.predict([weights[np.newaxis, ], trace_o[np.newaxis, ], trace_a[np.newaxis, ]])

        scal_q = np.dot(q_value[0], weights)
        action = np.argmax(scal_q)
        return action, scal_q

    # Given a bunch of experienced histories, update our models.
    def update_target(self, **kwargs):
        self.critic.update_target(**kwargs)

    def policy_update(self, update_actor=False):
        # np.random.seed(self.steps)
        ids, batch, _ = self.buffer.sample(self.batch_size)

        if self.direct_update:
            # Add recent experiences to the priority update batch
            batch = np.concatenate((batch, self.buffer.get(self.recent_experiences)), axis=0)
            ids = np.concatenate((ids, self.recent_experiences)).astype(int)

        # if self.dup is True, we train each sample in the batch on two
        # weight vectors, hence we duplicate the batch data
        if self.dup:
            batch = np.repeat(batch, 2, axis=0)
            ids = np.repeat(ids, 2, axis=0)
        
        # q_predict, q_target, w_batch, o_batch, a_last_batch
        q_predict, q_target, w_batch, o_batch, a_last_batch = self._get_training_data(batch)

        q_true = np.copy(q_predict)
        masks = np.zeros((len(batch), self.nb_action, self.qvalue_dim), dtype=float)
        
        for i, (_, action, reward, _, done, _, _) in enumerate(batch):
            q_true[i][action] = np.copy(reward)
            masks[i][action] = 1

            if not done:
                a_next = np.argmax(np.dot(q_target[i], w_batch[i]))
                q_true[i][action] += self.discount * q_target[i][a_next]

        dummy = q_true[:, 0, :]
        self.train_on_batch(w_batch, o_batch, a_last_batch, q_true, dummy, masks)

        loss = self.update_priorities(batch, ids)
        self.recent_experiences = []
    
        return loss

    def train_on_batch(self, w_batch, o_batch, a_last_batch, q_true, dummy=None, q_mask=None):
        self.critic.model.train_on_batch([w_batch, o_batch, a_last_batch], q_true)

    def _get_training_data(self, batch):

        o_batch, o_next_batch, a_last_batch, a_last_next_batch = self._get_full_states(batch)

        w_batch = self.get_training_weights(batch)

        # Predict both the model and target q-values
        predict_input = [w_batch, o_batch, a_last_batch]
        target_input = [w_batch, o_next_batch, a_last_next_batch]

        # Predict both the model and target q-values
        q_predict = self.critic.model.predict(predict_input)
        q_target = self.critic.target_model.predict(target_input)

        return q_predict, q_target, w_batch, o_batch, a_last_batch

    def update_priorities(self, batch, ids, ignore_dup=False, pr=False):
        """Given a batch of transitions, this method computes each transition's
        error and uses that error to update its priority in the replay buffer

        Arguments:
            batch {list} -- list of transitions
            ids {list} -- list of identifiers of each transition in the replay
                        buffer

        Returns:
            float -- The batch's mean loss
        """
        q_predict, q_target, w_batch, _, _ = self._get_training_data(batch)

        errors = np.zeros(len(batch))

        for i, (_, action, reward, _, terminal, _, _) in enumerate(batch):
            target = np.copy(reward)
                
            if not terminal:
                a_next = np.argmax(np.dot(q_target[i], w_batch[i]))
                target += self.discount * q_target[i][a_next]

            error = mae(q_predict[i][action], target)
            errors[i] = error

            # When dup is True, we train each sample on two weight vectors
            # Hence, there are two TD-errors per sample, we use the mean of
            # both errors to update the priorities
            if self.dup:
                if i % 2 == 0:
                    continue
                error = (error + errors[i - 1]) / 2
            self.buffer.update(ids[i], error)
            self.max_error = max(error, self.max_error)

        return np.mean(errors)

    def save_weights(self):
        """Saves the networks' weights to files identified by the agent's name
        and the current weight vector
        """
        self.critic.model.save_weights("output/networks/{}_dqn.weights".format(self.name))
            
    def save_model(self):
        """Saves the networks to files identified by the agent's name
        and the current weight vector
        """
        self.critic.model.save("output/networks/{}_dqn_model.h5".format(self.name))


class CN0Agent(CNAgent):
    def build_models(self):
        # Make only critic network (creates target model internally).
        self.critic = Critic0(self.nb_action, self.im_shape, self.nb_objective, self.qvalue_dim,
            timesteps=self.timesteps, net_type=self.net_type, memnn_size=self.memnn_size,
            lr=self.lr, action_conc=self.action_conc, action_embd=self.feature_embd, 
            weight_embd=self.feature_embd, backendtype=self.gpu_setting)

        with open("output/logs/info_"+self.extra+".log", "a") as f:
            f.write(time.asctime()+'\n\n')
            f.write(str(self.critic.model.get_config())+'\n\n')
            f.write(str(vars(self))+'\n\n')
        print(vars(self))

    def train_on_batch(self, w_batch, o_batch, a_last_batch, q_true, dummy=None, q_mask=None):
        self.critic.trainable_model.train_on_batch([w_batch, o_batch, a_last_batch, q_true, q_mask], [dummy, q_true])