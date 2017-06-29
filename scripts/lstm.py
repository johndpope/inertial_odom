from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

from hyperparams import *

# with tf.variable_scope('lstm'):
# tf Graph input
x = tf.placeholder("float", shape=[None, n_steps, n_input])
y = tf.placeholder("float", shape=[None, n_output])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_output]))
}

t = tf.unstack(x, n_steps, 1)
# Define a lstm cell with tensorflow
lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
# Get lstm cell output
outputs, states = rnn.static_rnn(lstm_cell, t, dtype=tf.float32)
# Linear activation, using rnn inner loop last output
pred= tf.matmul(outputs[-1], weights['out']) + biases['out']
cost=tf.nn.l2_loss(pred - y)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)