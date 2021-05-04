import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import *
from deep_sea_treasure import DeepSeaTreasure

def cal_states_similarity(state):

        input_t = tf.convert_to_tensor(state, dtype=np.float32)
        x = Lambda(lambda x: x / 255., name="input_normalizer")(input_t)

        x = TimeDistributed(Conv2D(filters=32, kernel_size=6, strides=2, 
                                    activation='relu', kernel_initializer='glorot_uniform',
                                    input_shape=x.shape))(x)
        x = TimeDistributed(MaxPool2D())(x)

        x = TimeDistributed(Conv2D(filters=64, kernel_size=5, strides=2, 
                                    activation='relu', kernel_initializer='glorot_uniform'))(x)
        x = TimeDistributed(MaxPool2D())(x)

        x = Flatten()(x)
        state = x.numpy()

        dist = np.linalg.norm(state[0, :]-state[1, :])
        
        # print(dist)
        return dist

# a = np.random.rand(2, 48, 48, 3)
# b = np.zeros((2, 48, 48, 3))
# state = np.concatenate((np.expand_dims(a, axis=0), 
#                                         np.expand_dims(b, axis=0)))
# score = cal_states_similarity(state)

# print(score)

env = DeepSeaTreasure(view=(5,5), full=True, scale=1)

o_t = env.reset()
terminal = False

env.show(plot=True)

while not terminal:
    a_t = int(input("0:left, 1:right, 2:up, 3:down | choose: "))
    o_t1, r_t, terminal, s_t1 = env.step(a_t)
    print(o_t1.shape)
    env.show(plot=True)

    o_t = o_t1
    print("Taking action", a_t, "at", s_t1["position"], "with reward", r_t)
env.reset()