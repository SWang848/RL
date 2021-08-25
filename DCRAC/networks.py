"""Classes of networks

Author: Xiaodong Nian
Time: October 2019
"""

import numpy as np
import tensorflow as tf
import keras.backend as K
from abc import ABC, abstractmethod

# Import various Keras tools.
from keras.models import Model, load_model
from keras.layers.recurrent import LSTM, GRU
from keras.layers import CuDNNLSTM, CuDNNGRU
from keras.layers import Dense, Input, Lambda, Flatten, Reshape
from keras.layers import Conv2D, LeakyReLU, MaxPooling2D, ReLU
from keras.layers import TimeDistributed, Masking, Dropout
from keras.layers.merge import Concatenate, Add
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomUniform
from keras.optimizers import Adam, SGD
# Self defined memory nerwork layer
from TemporalMemory import SimpleMemory

from config_agent import *



class Network(ABC):
    def __init__(self, 
                 nb_action=None,
                 observation_dim=None, 
                 nb_objective=None, 
                 q_dim=None, 
                 timesteps=1,
                 lr=1e-3,
                 tau=0.001,
                 clipnorm=1,
                 clipvalue=1,
                 nesterov=True,
                 momentum=0.9,
                 net_type='R',
                 memnn_size=10,
                 action_conc=ACTION_CONC,
                 action_embd=ACTION_INPUT_EMBD,
                 weight_embd=WEIGHT_INPUT_EMBD,
                 backendtype='1',
                 load_from_model=False,
                 model_path=None
                 ):
        
        if load_from_model and model_path is not None:
            self.model = load_model(model_path)
        
        else:
            self.nb_action = nb_action
            self.observation_dim = observation_dim
            self.nb_objective = nb_objective
            self.qvalue_dim = q_dim
            self.timesteps = timesteps
            self.lr = lr
            self.tau = tau
            self.clipnorm = clipnorm
            self.clipvalue = clipvalue
            self.momentum = momentum
            self.nesterov = nesterov

            self.normalize = True
            self.backendtype = backendtype

            self.action_conc = action_conc
            self.action_embd = action_embd
            self.weight_embd = weight_embd

            self.net_type = net_type
            self.memnn_size = memnn_size
            self.build_networks()

    @abstractmethod
    def build_networks(self):
        self.model = None
        self.target_model = None
        pass
    
    @abstractmethod
    def build_head(self):
        pass

    def build_base(self):
        """Builds the convolutional feature extraction component of the network

        Returns:
            tuple -- A tuple (Input, feature layer, weight layer) 
        """

        observation_input = Input((self.timesteps, )+self.observation_dim, name="observation_input")
        action_last_input = Input((self.timesteps, self.nb_action), name="action_last_input")
        weight_input = Input((self.nb_objective, ), name="weight_input")

        x = observation_input
        if self.normalize:
            x = Lambda(lambda x: x / 255., name="input_normalizer")(x)

        # Convolutional layers
        for c, (filters, kernel_size, strides) in enumerate(zip(CONV_FILTERS, CONV_KERNEL_SIZES, CONV_STRIDES)):
            x = TimeDistributed(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides), name="td_conv{}".format(c))(x)
            x = LeakyReLU(L_ALPHA, name="td_conv_leaky_relu{}".format(c))(x)
            x = TimeDistributed(MaxPooling2D(), name="td_conv_pooling{}".format(c))(x)
        
        if self.action_conc:
            action_embd_layer = action_last_input
            if self.net_type in ('R', 'M'):
                action_embd_layer = Masking(mask_value=-1, name="action_mask")(action_last_input)

            if self.action_embd:
                action_embd_layer = TimeDistributed(Dense(CONTEXT_SIZE, kernel_initializer=DENSE_INIT), 
                    name="action_embd")(action_embd_layer)
        
            encoded = TimeDistributed(Flatten(), name="td_conv_flat")(x)
            encoded = Concatenate(name="action_conc")([encoded, action_embd_layer])
        else:
            encoded = x

        # feature output
        if self.net_type == 'R':
            encoded = self.build_recurrent(encoded)
        elif self.net_type == 'M':
            encoded = self.build_memory_net(encoded)
        else: # net_type = F
            encoded = Dense(CONTEXT_SIZE, kernel_initializer=DENSE_INIT, name="post_conv_dense")(encoded)
            encoded = LeakyReLU(L_ALPHA)(encoded)
            encoded = Flatten()(encoded)

        # Connect the weight input
        feature_layer = self.build_feature(weight_input, encoded)
            
        return observation_input, action_last_input, weight_input, feature_layer
    
    def build_recurrent(self, inp):
        # double layer LSTM
        # feature_layer = self._rnn_layer(CONTEXT_SIZE, return_sequences=True, name="post_conv_lstm0")(inp)
        # feature_layer = self._rnn_layer(CONTEXT_SIZE, name="post_conv_lstm1")(feature_layer)

        # signle layer LSTM
        feature_layer = self._rnn_layer(CONTEXT_SIZE, name="post_conv_lstm")(inp)
        return feature_layer

    def build_memory_net(self, encoded):
        context = self._rnn_layer(CONTEXT_SIZE, kernel_initializer=DENSE_INIT, 
            return_sequences=True, name="post_conv_mem_context")(encoded)
        conc = Concatenate(name="conc")([encoded, context])
        memory = SimpleMemory(CONTEXT_SIZE, memory_size=self.memnn_size, name='o_t')(conc)

        last_context = Lambda(lambda x: x[:,-1,:], name='last_timestep_context')(context)
        output_layer = Dense(CONTEXT_SIZE, name='U_hxh_t')(last_context)
        # output_layer = LeakyReLU(L_ALPHA)(output_layer)
        # output_layer = ReLU(L_ALPHA)(output_layer)

        output_layer = Add(name='g_t')([output_layer, memory])
        output_layer = Dense(CONTEXT_SIZE, name='q_t')(output_layer)
        # output_layer = LeakyReLU(L_ALPHA)(output_layer)
        feature_layer = ReLU(L_ALPHA)(output_layer)

        # feature_layer = Dropout(rate=0.5)(output_layer)
        return feature_layer
    
    def build_feature(self, weight_input, feature_layer):
        if self.weight_embd:
            weight_embd = Dense(CONTEXT_SIZE, name='weight_embd', kernel_initializer=DENSE_INIT)(weight_input)
            return Concatenate(name="feature")([weight_embd, feature_layer])
        return Concatenate(name="feature")([weight_input, feature_layer])

    def update_target(self, soft_update=False):
        """Update target model. Copy main model weights to target network.
        
        Args:
            soft_update (bool, optional): [set to True to soft update target network by tau]. Defaults to False.
        """
        if not soft_update:
            self.target_model.set_weights(self.model.get_weights())
        else:
            self.target_model.set_weights(self.tau * np.array(self.model.get_weights())
                                  + (1 - self.tau) * np.array(self.target_model.get_weights()))

    def _rnn_layer(self, *args, **kwargs):
        if self.backendtype == '2':
            print("Using GPU setting LSTM.")
            return LSTM(*args, implementation=2, unroll=True, **kwargs)
            # return GRU(*args, implementation=2, unroll=True, **kwargs)
        elif self.backendtype == '3':
            print("Using CuDNNLSTM.")
            return CuDNNLSTM(*args, **kwargs)
            # return CuDNNGRU(*args, **kwargs)
        return LSTM(*args, **kwargs)
        # return GRU(*args, **kwargs)



class Actor(Network):
    def build_networks(self):
        """Builds the required networks, main and target policy networks

        Returns:
            tuple -- consisting of the main model, the target model
        """

        observation_input, action_last_input, weight_input, feature_layer = self.build_base()
        self.model = self.build_head(
            feature_layer, observation_input, action_last_input, weight_input)

        observation_input, action_last_input, weight_input, feature_layer = self.build_base()
        self.target_model = self.build_head(
            feature_layer, observation_input, action_last_input, weight_input)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=self.lr, clipnorm=self.clipnorm, clipvalue=self.clipvalue))

        self.model.summary()
        self.update_target()


    def build_head(self, feature_layer, observation_input, action_last_input, weight_input):

        head_dense = LeakyReLU(L_ALPHA)(Dense(
            ACTOR_FC_SIZE,
            name='actor_fc_0',
            kernel_initializer=DENSE_INIT)(feature_layer))
        head_dense = Dropout(0.3)(head_dense)
    
        for depth in range(1, ACTOR_FC_DEPTH):
            head_dense = LeakyReLU(L_ALPHA)(Dense(
                ACTOR_FC_SIZE,
                name='actor_fc_{}'.format(depth),
                kernel_initializer=DENSE_INIT)(head_dense))
            head_dense = Dropout(0.3)(head_dense)

        acts_prob = Dense(self.nb_action, name='acts_prob', 
            activation='softmax', kernel_initializer=DENSE_INIT)(head_dense)

        main_model = Model(inputs=[weight_input, observation_input, action_last_input], outputs=acts_prob)

        return main_model


    def choose_action(self, weight, observation, action_last, target=False, stochastic=True, return_probs=False):
        if target:
            acts_prob = self.target_model.predict([weight, observation, action_last])
        else:
            acts_prob = self.model.predict([weight, observation, action_last])

        action = np.random.choice(range(acts_prob.shape[1]), p=acts_prob.ravel()) if stochastic else np.argmax(acts_prob.ravel())

        if return_probs:
            return action, acts_prob.ravel()
        return action



class Critic(Network):
    def build_networks(self):
        """Builds the required networks, main and target q-value networks,
            a trainable (masked) main network

        Returns:
            tuple -- consisting of the main model, the target model, the 
                    trainable model and a predictive model
        """

        observation_input, action_last_input, weight_input, feature_layer = self.build_base()
        # Build dueling Q-value heads on top of the base
        self.model = self.build_head(feature_layer, observation_input, action_last_input, weight_input)

        observation_input, action_last_input, weight_input, feature_layer = self.build_base()
        self.target_model = self.build_head(feature_layer, observation_input, action_last_input, weight_input)

        self.model.compile(loss='mse', optimizer=SGD(
            lr=self.lr, clipnorm=self.clipnorm, clipvalue=self.clipvalue, 
            momentum=self.momentum, nesterov=self.nesterov))
        
        self.model.summary()
        self.update_target()


    def build_head(self, feature_layer, observation_input, action_last_input, weight_input):
        """Builds the Q-value head on top of the feature layer

        Arguments:
            feature_layer {Keras layer} -- The feature layer
            observation_input {Keras Input} -- The model's image input
            weight_input {Keras Layer} -- The model's weight features

        Returns:
            tuple -- Consisting of the main model, and a trainable model that
                        accepts a masked input
        """

        head_pred = self.build_dueling_head(feature_layer)

        y_pred = Reshape((self.nb_action, self.qvalue_dim))(head_pred)

        model = Model(inputs=[weight_input, observation_input, action_last_input], outputs=y_pred)

        return model
    

    def build_dueling_head(self, feature_layer):
        """Given a feature layer and weight input, this method builds the 
           Q-value outputs using a dueling architecture

        Returns:
            list -- List of outputs, one per action
        """

        # Build a dueling head with the required amount of outputs
        head_dense = self.build_dueling_dense(feature_layer)
        
        head_out = [
            Dense(
                self.qvalue_dim,
                name='dueling_out_value',
                activation='linear',
                kernel_initializer=DENSE_INIT)(head_dense[0]),
            Dense(
                self.nb_action * self.qvalue_dim,
                name='dueling_out_advantage',
                activation='linear',
                kernel_initializer=DENSE_INIT)(head_dense[1]),
        ]

        x = Concatenate(name="concat_heads")(head_out)
        x = Reshape((self.nb_action + 1, self.qvalue_dim))(x)

        # Dueling merge function
        outputs = [
            Lambda(lambda a: a[:, 0, :] + a[:, b + 1, :] - K.mean(a[:, 1:, :], axis=1, keepdims=False),
                   output_shape=(self.qvalue_dim, ))(x)
            for b in range(self.nb_action)
        ]

        return Concatenate(name="concat_outputs")(outputs)

    def build_dueling_dense(self, feature_layer):
        head_dense = [feature_layer] * 2

        for depth in range(DUELING_DEPTH):
            head_dense = [
                LeakyReLU(L_ALPHA)(Dense(
                    DUELING_LAYER_SIZE,
                    name='dueling_{}_{}'.format(depth, a),
                    kernel_initializer=DENSE_INIT)(head_dense[a]))
                for a in range(2)
            ]

        return head_dense



class Critic0(Critic):
    def build_networks(self):

        observation_input, action_last_input, weight_input, feature_layer = self.build_base()
        # Build dueling Q-value heads on top of the base
        self.model, self.trainable_model = self.build_head(
            feature_layer, observation_input, action_last_input, weight_input)

        observation_input, action_last_input, weight_input, feature_layer = self.build_base()
        self.target_model, _ = self.build_head(
            feature_layer, observation_input, action_last_input, weight_input)
        
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            'mse', # we only include this for the metrics
        ]

        self.trainable_model.compile(loss=losses, loss_weights=[1.,0.],
            optimizer=SGD(lr=self.lr, clipnorm=self.clipnorm, clipvalue=self.clipvalue, 
            momentum=self.momentum, nesterov=self.nesterov))
        
        self.model.summary()
        self.update_target()


    def build_head(self, feature_layer, observation_input, action_last_input, weight_input):
        """Builds the Q-value head on top of the feature layer

        Arguments:
            feature_layer {Keras layer} -- The feature layer
            observation_input {Keras Input} -- The model's image input
            weight_input {Keras Layer} -- The model's weight features

        Returns:
            tuple -- Consisting of the main model, and a trainable model that
                        accepts a masked input
        """

        head_pred = self.build_dueling_head(feature_layer)

        y_pred = Reshape((self.nb_action, self.qvalue_dim))(head_pred)

        # We mask the losses such that only losses of the relevant action
        # are taken into account for the network update
        y_true = Input(name='y_true', shape=(self.nb_action, self.qvalue_dim, ))

        mask = Input(name='mask', shape=(self.nb_action, self.qvalue_dim, ))

        loss_out = Lambda(Critic0.masked_error, output_shape=(1, ), name='loss')([y_true, y_pred, mask])

        trainable_model = Model(inputs=[weight_input, observation_input, action_last_input, y_true, mask], outputs=[loss_out, y_pred])
        main_model = Model(inputs=[weight_input, observation_input, action_last_input], outputs=y_pred)

        return main_model, trainable_model

    @staticmethod
    def masked_error(args):
        """
            Masked asolute error function

            Args:
                y_true: Target output
                y_pred: Actual output
                mask: Scales the loss, should be compatible with the shape of 
                        y_true, if an element of the mask is set to zero, the
                        corresponding loss is ignored
        """
        y_true, y_pred, mask = args
        loss = K.abs(y_true - y_pred)
        loss *= mask
        return K.sum(loss, axis=-2)



class ActorCritic(Critic):
    def __init__(self, *args, actor_lr=.25, **kwargs):
        self.actor_lr = actor_lr
        super(ActorCritic, self).__init__(*args, **kwargs)

    def build_networks(self):
        """Builds the required networks, main and target policy networks

        Returns:
            tuple -- consisting of the main model, the target model
        """

        observation_input, action_last_input, weight_input, feature_layer = self.build_base()
        self.model = self.build_head(feature_layer, observation_input, action_last_input, weight_input)

        observation_input, action_last_input, weight_input, feature_layer = self.build_base()
        self.target_model = self.build_head(feature_layer, observation_input, action_last_input, weight_input)

        self.model.compile(loss=['mae', 'categorical_crossentropy'], loss_weights=[1., self.actor_lr],
            optimizer=SGD(lr=self.lr, clipnorm=self.clipnorm, clipvalue=self.clipvalue, 
            momentum=self.momentum, nesterov=self.nesterov))

        self.model.summary()
        self.update_target()


    def build_actor_head(self, feature_layer):
        actor_head = feature_layer
        for depth in range(ACTOR_FC_DEPTH):
            # actor_head = LeakyReLU(L_ALPHA)(
            #     Dense(ACTOR_FC_SIZE, name='actor_fc_{}'.format(depth), kernel_initializer=DENSE_INIT
            #     )(actor_head))
            actor_head = Dense(ACTOR_FC_SIZE, name='actor_fc_{}'.format(depth), activation='tanh', kernel_initializer=DENSE_INIT)(actor_head)
            actor_head = Dropout(0.2)(actor_head)

        acts_prob = Dense(self.nb_action, name='acts_prob', 
            activation='softmax', kernel_initializer=DENSE_INIT)(actor_head)
        return acts_prob


    def build_head(self, feature_layer, observation_input, action_last_input, weight_input):

        # ACTOR
        acts_prob = self.build_actor_head(feature_layer)

        # CRITIC
        critic_head = self.build_dueling_head(feature_layer)
        y_pred = Reshape((self.nb_action, self.qvalue_dim))(critic_head)

        # MODEL
        model = Model(inputs=[weight_input, observation_input, action_last_input], outputs=[y_pred, acts_prob])
        return model


    def choose_action(self, weight, observation, action_last_input, target=False, stochastic=True, return_probs=False):
        if target:
            acts_prob = self.target_model.predict([weight, observation, action_last_input])[1]
        else:
            acts_prob = self.model.predict([weight, observation, action_last_input])[1]

        action = np.random.choice(range(acts_prob.shape[1]), p=acts_prob.ravel()) if stochastic else np.argmax(acts_prob.ravel())

        if return_probs:
            return action, acts_prob.ravel()
        return action
    
    


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        # prob = K.sum(y_true * y_pred, axis=-1)
        # old_prob = K.sum(y_true * old_prediction, axis=-1)
        rho = prob / (old_prob + 1e-10)
        return -K.mean(K.minimum(rho * advantage, K.clip(rho, min_value=1-CLIPPING_LOSS_RATIO, max_value=1+CLIPPING_LOSS_RATIO) * advantage) 
                        + ENTROPY_LOSS_RATIO * (prob * K.log(prob + 1e-10)))
        # return -K.mean(K.minimum(rho * advantage, K.clip(rho, min_value=1-CLIPPING_LOSS_RATIO, max_value=1+CLIPPING_LOSS_RATIO) * advantage) 
        #                  * K.log(prob + 1e-10))
    return loss


class ActorCriticER(ActorCritic):
    def build_networks(self):
        """Builds the required networks, main and target policy networks

        Returns:
            tuple -- consisting of the main model, the target model
        """
        objective = Input(shape=(1, ), name="Objective")
        old_probs = Input(shape=(self.nb_action,), name="Old_Probs")

        observation_input, action_last_input, weight_input, feature_layer = self.build_base()
        self.model, self.trainable_model = self.build_head(feature_layer, observation_input, action_last_input, weight_input, objective, old_probs)

        observation_input, action_last_input, weight_input, feature_layer = self.build_base()
        self.target_model, _ = self.build_head(feature_layer, observation_input, action_last_input, weight_input, objective, old_probs)

        self.trainable_model.compile(loss=['mae', proximal_policy_optimization_loss(objective, old_probs)], loss_weights=[1.,self.actor_lr],
            optimizer=Adam(lr=self.lr))
            # optimizer=SGD(lr=self.lr, clipnorm=self.clipnorm, clipvalue=self.clipvalue, momentum=self.momentum, nesterov=self.nesterov))

        self.model.summary()
        self.update_target()


    def build_head(self, feature_layer, observation_input, action_last_input, weight_input, objective, old_probs):
        # ACTOR
        acts_prob = self.build_actor_head(feature_layer)
        # CRITIC
        critic_head = self.build_dueling_head(feature_layer)
        y_pred = Reshape((self.nb_action, self.qvalue_dim))(critic_head)
        # MODEL
        model = Model(inputs=[weight_input, observation_input, action_last_input], outputs=[y_pred, acts_prob])
        trainable_model = Model(inputs=[weight_input, observation_input, action_last_input, objective, old_probs], outputs=[y_pred, acts_prob])
        return model, trainable_model