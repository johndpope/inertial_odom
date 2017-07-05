from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from hyperparams import *
from scripts.helper_funcs import *
from scripts.utils import *
import tensorflow.contrib.slim as slim
from scripts.losses import rtLoss
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

	with slim.arg_scope([slim.fully_connected],
						activation_fn=None,
						# normalizer_fn=slim.batch_norm,
						weights_initializer=\
						tf.truncated_normal_initializer(stddev=0.01),
						weights_regularizer=slim.l2_regularizer(0.0005)):
						feat = slim.fully_connected(tf.reshape(x,[batch_size*n_steps,n_input]), 128)
						feat = tf.reshape(feat,[batch_size,n_steps,128])

	# x is B x n_steps x 6
	t = tf.unstack(feat, n_steps, axis=1)
	# t is n_steps different features, that are each B x 6
	# Define a lstm cell with tensorflow
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
	# lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=dropout)
	# Get lstm cell output
	outputs, states = rnn.static_rnn(lstm_cell, t, dtype=tf.float32)
	# Linear activation, using rnn inner loop last output
	pred= tf.matmul(outputs[-1], weights['out']) + biases['out']
	
	(d_roll, d_pitch, d_yaw)=quat_to_euler(pred[:,3:7])
	d_x=pred[:,0]
	d_y=pred[:,1]
	d_z=pred[:,2]
	Tv12=pose2mat(y)
	Ti12=pose2mat(pred)
	#RT Loss for predicted 
	rtd, rta =rtLoss(Ti12,Tv12)

	#rtd_z, rta_z =rtLoss(Tv12,Tv12)

	with tf.name_scope('predicted'):
		#predicted translation
		with tf.name_scope('translation'):
			d_x_=tf.summary.histogram("x", d_x)
			d_y_=tf.summary.histogram("y", d_y)
			d_z_=tf.summary.histogram("z", d_z)
		#predicted quaternion
		with tf.name_scope('quaternion'):
			qw_=tf.summary.histogram("qw", pred[:,3])
			qx_=tf.summary.histogram("qx", pred[:,4])
			qy_=tf.summary.histogram("qy", pred[:,5])
			qz_=tf.summary.histogram("qz", pred[:,6])
		#predicted euler angles
		with tf.name_scope('euler'):
			d_roll_=tf.summary.histogram("roll", d_roll)
			d_pitch_=tf.summary.histogram("pitch", d_pitch)
			d_yaw_=tf.summary.histogram("yaw", d_yaw)
	#losses
	pos_loss=tf.nn.l2_loss(pred[:,0:3] - y[:,0:3])/batch_size 
	rot_loss=tf.nn.l2_loss(pred[:,3:7] - y[:,3:7])/batch_size
	cost = (rtd+rta)
	# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
if (do_zero_motion):
	with tf.name_scope('inputs'):
		with tf.name_scope('angular_velocity'):
			w_x=tf.summary.histogram("x", x[:,:,0])
			w_y=tf.summary.histogram("y", x[:,:,1])
			w_z=tf.summary.histogram("z", x[:,:,2])
		with tf.name_scope('linear_acceleration'):
			accel_x=tf.summary.histogram("x", x[:,:,3])
			accel_y=tf.summary.histogram("y", x[:,:,4])
			accel_z=tf.summary.histogram("z", x[:,:,5])
	(d_roll, d_pitch, d_yaw)=quat_to_euler(y[:,3:7])
	d_x=y[:,0]
	d_y=y[:,1]
	d_z=y[:,2]

	Tv12=pose2mat(y)
	T_z=tf.tile(tf.reshape(tf.Variable([[[1,0,0,0],
				 [0,1,0,0],
				 [0,0,1,0],
				 [0,0,0,1]]], dtype=tf.float32),[1,4,4]),[batch_size,1,1])
	#ground truth RT Loss
	rtd, rta = rtLoss(T_z,Tv12)
	with tf.name_scope('ground_truth'):
		#ground truth translation
		with tf.name_scope('translation'):
			d_x_=tf.summary.histogram("x", d_x)
			d_y_=tf.summary.histogram("y", d_y)
			d_z_=tf.summary.histogram("z", d_z)
		#ground truth euler angles
		with tf.name_scope('euler'):
			d_roll_=tf.summary.histogram("roll", d_roll)
			d_pitch_=tf.summary.histogram("pitch", d_pitch)
			d_yaw_=tf.summary.histogram("yaw", d_yaw)
		#ground truth quaternion 
		with tf.name_scope('quaternion'):
			qw_=tf.summary.histogram("qw", y[:,3])
			qx_=tf.summary.histogram("qx", y[:,4])
			qy_=tf.summary.histogram("qy", y[:,5])
			qz_=tf.summary.histogram("qz", y[:,6])

	zero_pos= tf.zeros([batch_size,3], tf.float32)
	zero_rot= tf.tile(tf.reshape(tf.Variable([[1,0,0,0]], dtype=tf.float32),[1,4]),[batch_size,1])
	pos_loss = tf.nn.l2_loss(zero_pos- y[:,0:3])/batch_size
	rot_loss = tf.nn.l2_loss(zero_rot- y[:,3:7])/batch_size
	cost=(rtd+rta)
with tf.name_scope('l2_loss'):
	pos_loss_summ = tf.summary.scalar('translation_loss',pos_loss)
	rot_loss_summ = tf.summary.scalar('quaternion_loss',rot_loss)
	#cost_summ = tf.summary.scalar('loss',cost)
with tf.name_scope('RT_loss'):
	cost_summ = tf.summary.scalar('loss',cost)
	rtd_e=tf.summary.scalar('rtd', rtd)
	rta_e=tf.summary.scalar('rta', rta)
	#cross check zeros
	#rtd_z=tf.summary.scalar('rtd_z', rtd_z)
	#rta_z=tf.summary.scalar('rta_z', rta_z)
summary = tf.summary.merge_all()

