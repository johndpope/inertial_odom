from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

from hyperparams import *

with tf.variable_scope('lstm'):
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])

    # Define weights
    weights = {
        'hidden1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
        'hidden2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
        'hidden3': tf.Variable(tf.random_normal([n_hidden2, n_hidden3])),
        'hidden4': tf.Variable(tf.random_normal([n_hidden3, n_hidden4])),
        'out': tf.Variable(tf.random_normal([n_hidden4, n_output])),
    }
    biases = {
        'hidden1': tf.Variable(tf.random_normal([n_hidden1])),
        'hidden2': tf.Variable(tf.random_normal([n_hidden2])),
        'hidden3': tf.Variable(tf.random_normal([n_hidden3])),
        'hidden4': tf.Variable(tf.random_normal([n_hidden4])),
        'out': tf.Variable(tf.random_normal([n_output])),
    }

    # x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden1, forget_bias=1.0)

    # Get lstm cell output
    h_o1, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    #second hidden layer
    h_in2=tf.matmul(h_o1[-1], weights['hidden1']) + biases['hidden1']
    h_o2=tf.nn.relu(h_in2)
    #third hidden layer
    h_in3=tf.matmul(h_o2[-1], weights['hidden2']) + biases['hidden2']
    h_o3=tf.nn.relu(h_in3)
    #fourth hidden layer
    h_in4=tf.matmul(h_o3[-1], weights['hidden3']) + biases['hidden3']
    h_o4=tf.nn.relu(h_in4)
    #output layer
    h_out= tf.matmul(h_o4[-1], weights['hidden4']) + biases['hidden4']
    pred = tf.nn.relu(h_out)
    # Define loss and optimizer
    cost=tf.nn.l2_loss(pred - y)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)