import tensorflow as tf

sess = tf.Session()
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from tensorflow.keras import initializers
from collections import deque
from tensorflow.keras.layers import Input, Dense, Conv2D, Lambda, GRU, Concatenate, Dot
from tensorflow.keras.layers import Flatten, BatchNormalization, Reshape, UpSampling2D, LeakyReLU
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop, Optimizer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
import numpy as np
import random
import pickle
import keras_contrib as KC

def swish(x):
    return K.sigmoid(x) * x
    
class MIRAGEAgent:
    def __init__(self, state_shape, action_shape):
        self.state_size = state_shape
        self.action_size = action_shape
        self.memory = deque(maxlen=50)

        self.max_len = 50
        self.actor_editable_memory = np.zeros((1, self.max_len))
        self.lr = 0.001
        self.gf = 4
        self.df = 4
        self.optimizer = Adam(lr=self.lr)
        self.actor = self.build_generator()
        self.critic = self.build_discriminator()
        
        self.actor_loss = 'msle'
        self.critic_loss = 'msle'
        self.prev_action = np.zeros((1, self.action_size))
        self.list_rewards = [0.01]
        
    def build_generator(self):
        def conv2d_head(x, filters, stride, f_size=4, activation=swish, action_size=None):
            x = tfp.layers.Convolution2DFlipout(filters, f_size, stride, padding='same', activation=activation)(x)
            return x
        
        def rnn_body(x, units, activation=swish, reshape=True, stateful=True):
            shape = tf.keras.backend.int_shape(x)

            if reshape:
                x = tf.keras.layers.Reshape((shape[1], shape[2]))(x)
            x = tf.keras.layers.LSTM(units, activation=activation, stateful=stateful)(x)
            return x
        
        def capsule_tail(x, action_size):
            x_shape = K.int_shape(x)
            x = Reshape((1, 1, x_shape[1]))(x)
            x = KC.layers.Capsule(action_size, 10, 4, False)(x)
            x = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(x)
            return x

        #Batch shape for stateful rnn.
        inp = Input(batch_shape=(1,)+self.state_size)
        x = conv2d_head(inp, 4, 3)
        x = conv2d_head(x, 8, 2)
        
        x = conv2d_head(x, 16, 2)
        x = conv2d_head(x, 32, 1)
        x = conv2d_head(x, 1, 1)

        x = rnn_body(x, self.action_size // 2)

        output_act = capsule_tail(x, self.action_size)

        model = Model(inp, output_act)
        model.summary()
        return model
 
    def build_discriminator(self):
        def conv2d(layer_input, filters, stride, f_size=4, activation=swish, bn=True, action_size=None):
            x = tfp.layers.Convolution2DFlipout(filters, f_size, stride, padding='same', activation=activation)(layer_input)
            return x
        
        def tail(x, action_size):
            x_shape = K.int_shape(x)
            x = Reshape((1, 1, x_shape[1]))(x)
            x = KC.layers.Capsule(action_size, 10, 4, False)(x)
            x = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(x)
            
            return x

        inp = Input(shape=self.state_size)
        x = conv2d(inp, self.df, 8)
        x = conv2d(x, self.df * 2, 4)
        output_crit = tail(x, 1)

        model = Model(g0, output_crit)
        model.summary()
        return model
        
    def remember(self, prev_state, action, reward, state, confidence):
        self.memory.append((prev_state,
                            action,
                            reward,
                            state,
                            confidence))

    def act(self, state):
        choice_matrix = self.actor.predict(state)
        return choice_matrix

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        actor_loss = []
        

        for prev_state, action, reward, state, action_prob in minibatch:
            advantage = np.add(np.zeros((1,self.action_size)), 0.0001) #To minimize risk of nans.
            target = np.add(np.zeros((1,1)), 0.0001)

            prev_value = self.critic.predict(prev_state)
            value = self.critic.predict(state)
            confidence = np.amax(action_prob)
            
                            
            actor_reward = reward
            print("\nActor reward:", actor_reward)
            advantage[0][action] = actor_reward * (value) - prev_value
            
            target[0][0] = actor_reward * value
            
            actor_loss_hist = self.actor.fit(state, advantage, epochs=1, verbose=0)
            critic_loss_hist = self.critic.fit(state, target, epochs=1, verbose=0)
            
            actor_loss += actor_loss_hist.history['loss']
            
            
        
        avg_loss = np.average(actor_loss)
        rew = avg_loss
        return rew

    def load(self, actor_mod, critic_mod, mem):
        self.actor.load_weights(actor_mod)
        with open(mem, 'rb') as load_mem:
            self.memory = pickle.load(load_mem)
    def save(self, actor_mod, critic_mod, mem):
        self.actor.save_weights(actor_mod)
        with open(mem, 'wb') as saved_mem:
            pickle.dump(self.memory, saved_mem)
        

    def initialize(self):
        self.actor.compile(loss=self.actor_loss, optimizer = self.optimizer)
        self.critic.compile(loss=self.critic_loss, optimizer = self.optimizer)

    def get_weights(self):
        x = np.array(self.actor.get_weights())
        x = np.reshape(x, (1,x.shape[0]))
        return x
    
if __name__ == '__main__' :
    mirage = MIRAGEAgent((1920,1080,3), 1920)
