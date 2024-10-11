import tensorflow as tf
import keras
import keras.api.backend as k
import keras.api.layers as kl
import keras.api.activations as ka

import numpy as np


from abc import ABC, abstractmethod

class BaseEncoder(keras.Model, ABC):
    def __init__(self, sampling_rate=48000, fps=60, n_frame=4):
        super(BaseEncoder, self).__init__()
        self.sampling_rate = sampling_rate
        self.FPS = fps
        self.n_frame = n_frame

    def call(self, x):
        # left side
        left = x[:, 0, :]
        left = self.encode_single_channel(left)
        # right side
        right = x[:, 1, :]
        right = self.encode_single_channel(right)
        return tf.concat((left, right), axis=len(left.shape) - 1)

    @abstractmethod
    def encode_single_channel(self, data):
        pass


class Conv1dEncoder(BaseEncoder):
    def __init__(self, sampling_rate=48000, fps=60, n_frame=1):
        super(Conv1dEncoder, self).__init__(sampling_rate, fps, n_frame)
        self.num_to_subsample = 8
        self.num_samples = (self.sampling_rate / self.FPS) * self.n_frame
        assert int(self.num_samples) == self.num_samples

        # self.pool = tf.keras.layers.MaxPool1D()
        self.pool = kl.MaxPool1D()
        self.conv1 = kl.Conv1D(16, 16, strides=8)
        # self.conv1 = tf.keras.layers.Conv1D(16, 16, strides=8)

    def encode_single_channel(self, data):
        x = data[:, ::self.num_to_subsample]
        x = x[:, :, None]
        x = tf.nn.relu(self.conv1(x))
        x = self.pool(x)
        return x

class Model(keras.Model):
    def __init__(self, num_actions):
        super().__init__(name='basic_dqn')
        self.encoder = Conv1dEncoder()
        # self.flatten = tf.keras.layers.Flatten()
        self.flatten = kl.Flatten()
        self.fc1 = kl.Dense(256, activation='relu', kernel_initializer="he_uniform")
        self.fc2 = kl.Dense(256, activation='relu', kernel_initializer="he_uniform")
        self.action_layer = kl.Dense(num_actions)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.action_layer(x)
        return x

    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action[0], q_values[0]
