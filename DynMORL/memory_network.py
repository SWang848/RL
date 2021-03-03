import keras.backend as K
import numpy as np
from tensorflow import keras
import tensorflow as tf
import keras

class MemoryNetwork(keras.layers.recurrent.RNN):
    def __init__(self, units, memory_size=10, **kwargs):
        cell = MemoryCell(units, memory_size)
        super(MemoryNetwork, self).__init__(cell=cell, **kwargs)

    def call(self, inputs):
        return super(MemoryNetwork, self).call(inputs)

class MemoryCell(keras.layers.Layer):
    def __init__(self, units, memory_size, **kwargs):
        self.units = units
        self.memory_size = memory_size
        self.state_size = (self.units, ) * (self.memory_size*2)
        print(self.state_size)
        super(MemoryCell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # w_init = tf.random_uniform_initializer()
        # self.W_value = tf.Variable(
        #     initial_value=w_init(shape=(input_dim-self.units, self.units), dtype="float32"),
        #     trainable=True
        # )
        # self.W_key = tf.Variable(
        #     initial_value=w_init(shape=(input_dim-self.units, self.units), dtype="float32"),
        #     trainable=True
        # )
        self.W_key = self.add_weight(
            shape=(input_dim - self.units, self.units),
            name='W_key',
            initializer=keras.initializers.glorot_uniform())

        self.W_value = self.add_weight(
            shape=(input_dim - self.units, self.units),
            name='W_value',
            initializer=keras.initializers.glorot_uniform())
        super(MemoryCell, self).build(input_shape)

    def call(self, inputs, states):
        e_t = inputs[:, :-self.units]
        h_t = inputs[:, -self.units:]

        M_key = list(states[:self.memory_size])
        M_value = list(states[self.memory_size:])

        M_key_tensor = K.stack(M_key, axis=1)
        M_value_tensor = K.stack(M_value, axis=1)

        at = K.softmax(K.batch_dot(M_key_tensor, h_t, axes=[2,1]))
        output = K.batch_dot(M_value_tensor, at, axes=[1,1])

        M_key.pop(0)
        M_value.pop(0)
        m_key = K.dot(e_t, self.W_key)
        m_value = K.dot(e_t, self.W_value)
        M_key.append(m_key)
        M_value.append(m_value)

        return output, M_key+M_value       
