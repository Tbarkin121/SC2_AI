import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

class QNet(tf.keras.Model):
  """Actor network."""
  def __init__(self, num_actions: int, num_hidden_units: int):
    """Initialize."""
    super().__init__()
    # self.actor_input = layers.Input(shape=(4))
    self.d1 = layers.Dense(num_hidden_units)
    self.lr1 = layers.LeakyReLU()
    self.d2 = layers.Dense(num_hidden_units)
    self.lr2 = layers.LeakyReLU()
    # self.a = layers.Dense(num_actions, activation='tanh')
    # self.a = layers.Dense(num_actions, activation='sigmoid')
    self.Q = layers.Dense(num_actions)
    

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.d1(inputs)
    # x = tf.keras.activations.tanh(x)
    x = self.lr1(x)
    x = self.d2(x)
    x = self.lr2(x)
    return self.Q(x)

class QLearner:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.90, e_greedy=0., load_model = False):
        self.actions = actions  # a list
        self.num_actions = len(actions)
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        if(load_model):
            self.q_net = tf.keras.models.load_model('saved_model/mineral_walker')
        else:
            self.q_net = QNet(self.num_actions, 128)
            self.q_net.compile()
        self.q_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.target_net = QNet(self.num_actions, 128)
        self.target_net.compile()
        self.learn_steps = 0
        self.copy_rate = 10

    def choose_action(self, observation):
        observation = tf.expand_dims(tf.Variable(observation, dtype=tf.float32), 0)
        
        if np.random.uniform() < self.epsilon:
            # choose best action
            q_val = self.q_net(observation)
            action = tf.argmax(q_val, axis=1)

            # probs = tf.nn.softmax(q_val)
            # log_probs = tf.math.log(probs)
            # action = tf.random.categorical(log_probs, 1)[:,0]
            # action = tf.constant([0])
        else:
            # choose random action
            action = tf.constant([np.random.choice(self.actions)])
            # action = tf.constant([0])
        return action

    def learn(self, s1, a, r, s2):
        print('r = {}'.format(r))
        s1 = tf.expand_dims(tf.Variable(s1), 0)
        s2 = tf.expand_dims(tf.Variable(s2), 0)
        # print('s1 = {}'.format(s1))
        # print('s2 = {}'.format(s2))
        a = tf.expand_dims(tf.Variable(a), 0)
        with tf.GradientTape() as tape:
            print('a = {}'.format(a))
            q1 = self.q_net(s1, training=True)
            q2 = self.target_net(s2, training=False)

            print('q1 = {}'.format(q1))
            print('q2 = {}'.format(q2))
            q2_max_val = tf.math.reduce_max(q2)
            print('q2_max_val = {}'.format(q2_max_val))
            q1_selected = tf.gather(q1, a, batch_dims=1)
            print('q1_selected = {}'.format(q1_selected))
            # q_target = tf.gather(self.q_net(s), a, axis=1, batch_dims=1)[0][0]
            q_target = r + self.gamma * q2_max_val
            print('q_target = {}'.format(q_target))
            # loss = (q_target - q1_selected)**2
            loss = 0.5*(q1_selected - q_target)**2
            # loss = tf.math.reduce_sum( (q_target - q_predict)**2)
            print('loss = {}'.format(loss))

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        # print('grads = {}'.format(grads))
        self.q_opt.apply_gradients(zip(grads, self.q_net.trainable_variables))
        
        if(self.learn_steps % self.copy_rate == 0):
            print('target weghts updated')
            self.target_net.set_weights(self.q_net.get_weights()) 
        self.learn_steps += 1
        # print(self.learn_steps)
        # print_q = self.q_net(s)
        # print("\033[H\033[J") 
        # print('q_values for state s = {}'.format(s))
        # print(print_q.numpy()[0])

   