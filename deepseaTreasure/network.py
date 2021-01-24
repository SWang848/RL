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

from history import *
from config_agent import *


def LEAKY_RELU(): return LeakyReLU(0.01)

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
class Network:

    def __init__(self, 
                 obj_cnt,
                 action_count,
                 scale, 
                 lstm, 
                 alg,
                 lr,
                 clipnorm,
                 clipvalue,
                 momentum,
                 nesterov,
                 input_shape):

        # self.frames_per_state = frames_per_state,
        # self.im_shape = *im_shape,
        self.obj_cnt = obj_cnt
        self.action_count = action_count
        self.scale = scale
        self.lstm = lstm
        self.alg = alg
        self.lr = lr
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.momentum = momentum
        self.nesterov = nesterov
        self.input_shape = input_shape
        
    def build_base(self):
        """Builds the feature extraction component of the network

        Returns:
            tuple -- A tuple (Input, feature layer, weight layer) 
        """

        state_input = Input(self.input_shape)

        # We always build the weight input, but don't connect it to the network
        # when it is not required.
        
        weight_input = Input((self.obj_cnt, ), name="weight_input")
        
        x = Lambda(lambda x: x / 255., name="input_normalizer")(state_input)

        # Convolutional layers
        for c, (
                filters, kernel_size, strides
        ) in enumerate(zip(CONV_FILTERS, CONV_KERNEL_SIZES, CONV_STRIDES)):
            x = TimeDistributed(
                Conv2D(
                    filters=int(filters / self.scale),
                    kernel_size=kernel_size,
                    strides=strides,
                    name="conv{}".format(c)))(x)
            x = LEAKY_RELU()(x)
            x = TimeDistributed(MaxPooling2D())(x)

        # Per dimension dense layer
        x = Dense(
            int(POST_CONV_DENSE_SIZE / self.scale),
            kernel_initializer=DENSE_INIT,
            name="post_conv_dense")(x)
        x = LEAKY_RELU()(x)
        

        # if self.lstm:
        #     flatten_layer = TimeDistributed(Flatten())(x)
        #     lstm_layer, _ = LSTM(256, return_sequences=True, dropout=0.5)(flatten_layer)
        #     print(lstm_layer.shape)
        #     feature_layer = Dense(512)(lstm_layer)
        # else:
        #     feature_layer = Flatten()(x)

        feature_layer = TimeDistributed(Flatten())(x)
        feature_layer = Dense(512)(feature_layer)

        self.shared_length = 0

        return state_input, feature_layer, weight_input

    def build_dueling_head(self, feature_layer, weight_input, obj_cnt,
                            per_stream_dense_size):
        """Given a feature layer and weight input, this method builds the
            Q-value outputs using a dueling architecture

        Returns:
            list -- List of outputs, one per action
        """

        weight_embedding = Embedding(1, 512, input_length=2)(weight_input)
        multiplied = Multiply()([weight_embedding, feature_layer])
        features = Flatten()(multiplied)

        # Connect the weight input only if in conditionned network mode
        # features = Concatenate(name="features")(
        #     [weight_input,
        #         feature_layer]) if ("cond" in self.alg or "uvfa" in self.alg) else feature_layer

        # Build a dueling head with the required amount of outputs
        head_dense = [
            LEAKY_RELU()(Dense(
                per_stream_dense_size,
                name='dueling_0_{}'.format(a),
                kernel_initializer=DENSE_INIT)(features))
            for a in range(2)
        ]

        for depth in range(1, DUELING_DEPTH):
            head_dense = [
                LEAKY_RELU()(Dense(
                    per_stream_dense_size,
                    name='dueling_{}_{}'.format(depth, a),
                    kernel_initializer=DENSE_INIT)(head_dense[a]))
                for a in range(2)
            ]

        head_out = [
            Dense(
                obj_cnt,
                name='dueling_out_{}'.format(a),
                activation='linear',
                kernel_initializer=DENSE_INIT)(head_dense[0] if a == 0 else head_dense[1])
            for a in range(self.action_count + 1)
        ]

        x = Concatenate(name="concat_heads")(head_out)

        x = Reshape((self.action_count + 1, obj_cnt))(x)

        # Dueling merge function
        outputs = [
            Lambda(lambda a: a[:, 0, :] + a[:, b + 1, :] -
                    K.mean(a[:, 1:, :], axis=1, keepdims=False),
                    output_shape=(obj_cnt, ))(x)
            for b in range(self.action_count)
        ]

        return Concatenate(name="concat_outputs")(outputs)

    def build_head(self, feature_layer, inp, weight_input):
        """Builds the Q-value head on top of the feature layer

        Arguments:
            feature_layer {Keras layer} -- The feature layer
            inp {Keras Input} -- The model's image input
            weight_features {Keras Layer} -- The model's weight features

        Returns:
            tuple -- Consisting of the main model, and a trainable model that
                        accepts a masked input
        """

        head_pred = self.build_dueling_head(feature_layer, weight_input,
                                            self.qvalue_dim(), DUELING_LAYER_SIZE)

        y_pred = Reshape((self.action_count,  self.qvalue_dim()))(head_pred)

        # We mask the losses such that only losses of the relevant action
        # are taken into account for the network update, based on:
        # https://github.com/keras-rl/keras-rl/blob/master/rl/agents/sarsa.py
        y_true = Input(name='y_true', shape=(
            self.action_count,  self.qvalue_dim(), ))

        mask = Input(name='mask', shape=(
            self.action_count,  self.qvalue_dim(), ))

        loss_out = Lambda(
            masked_error, output_shape=(1, ),
            name='loss')([y_true, y_pred, mask])

        trainable_model = Model([weight_input, inp, y_true, mask],
                                [loss_out, y_pred])
        main_model = Model([weight_input, inp], y_pred)

        return main_model, trainable_model

    def build_networks(self):
        """Builds the required networks, main and target q-value networks,
            a trainable (masked) main network and a predictive network

        Returns:
            tuple -- consisting of the main model, the target model, the 
                    trainable model and a predictive model
        """
        state_input, feature_layer, weight_input = self.build_base()

        # Build dueling Q-value heads on top of the base
        main_model, trainable_model = self.build_head(
            feature_layer, state_input, weight_input)

        state_input, feature_layer, weight_input = self.build_base()
        target_model, _ = self.build_head(
            feature_layer, state_input, weight_input)

        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            # we only include this for the metrics
            lambda y_true, y_pred: K.zeros_like(y_pred),
        ]

        trainable_model.compile(
            loss=losses,
            optimizer=SGD(
                lr=self.lr,
                clipnorm=self.clipnorm,
                clipvalue=self.clipvalue,
                momentum=self.momentum,
                nesterov=self.nesterov))

        return main_model, target_model, trainable_model

    def qvalue_dim(self):
        return 1 if self.has_scalar_qvalues() else self.obj_cnt

    def has_scalar_qvalues(self):
        return "scal" in self.alg or "uvfa" in self.alg