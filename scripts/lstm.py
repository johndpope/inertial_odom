from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from hyperparams import *
from scripts.helper_funcs import *

# with tf.variable_scope('lstm'):
# tf Graph input
x = tf.placeholder("float", shape=[batch_size, n_steps, n_input])
y = tf.placeholder("float", shape=[batch_size, n_output])
if not (do_zero_motion):
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
	pos_loss=tf.nn.l2_loss(pred[:,0:3] - y[:,0:3])/batch_size 
	rot_loss=tf.nn.l2_loss(pred[:,3:7] - y[:,3:7])/batch_size
	cost = (pos_loss+rot_loss)
	# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
if (do_zero_motion):
	(d_roll, d_pitch, d_yaw)=quat_to_euler(y[:,3:7])
	d_x=y[:,0]
	d_y=y[:,1]
	d_z=y[:,2]
	d_x_=tf.summary.histogram("dx", d_x)
	d_y_=tf.summary.histogram("dy", d_y)
	d_z_=tf.summary.histogram("dz", d_z)
	d_roll_=tf.summary.histogram("droll", d_roll)
	d_pitch_=tf.summary.histogram("dpitch", d_pitch)
	d_yaw_=tf.summary.histogram("dyaw", d_yaw)

	zero_pos= tf.zeros([batch_size,3], tf.float32)
	zero_rot= tf.tile(tf.reshape(tf.Variable([[1,0,0,0]], dtype=tf.float32),[1,4]),[batch_size,1])
	pos_loss = tf.nn.l2_loss(zero_pos- y[:,0:3])/batch_size
	rot_loss = tf.nn.l2_loss(zero_rot- y[:,3:7])/batch_size
	cost=(pos_loss+rot_loss)

pos_loss_summ = tf.summary.scalar('translation_loss',pos_loss)
rot_loss_summ = tf.summary.scalar('quaternion_loss',rot_loss)
cost_summ = tf.summary.scalar('loss', cost)
summary = tf.summary.merge_all()

