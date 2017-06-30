import tensorflow as tf
from io_libs.reader import read_tfrecords
import os
from scripts.lstm import *
import numpy as np
from io_libs.writer import *
from scripts.hyperparams import *
from scripts.helper_funcs import *
from scripts.losses import rtLoss
if __name__=="__main__":
	if(create_tfrecords):
		write_tfrecords()
	nRecords= len([name for name in os.listdir(input_Dir) if os.path.isfile(os.path.join(input_Dir, name))])
	print 'found %d records' % nRecords
	train_dataset = [input_Dir+'/sample_'+ str(i)+'.tfrecord' for i in range(1,int(nRecords*0.75))]
	validation_dataset = [input_Dir+'/sample_'+ str(i)+'.tfrecord' for i in range(int(nRecords*0.75),nRecords)]
	train_data = tf.train.string_input_producer(train_dataset, shuffle=True)
	validation_data = tf.train.string_input_producer(validation_dataset, shuffle=True)
	imu_t,rel_pose_t,imu_len_t=read_tfrecords(train_data)
	imu_t,rel_pose_t,imu_len_t = tf.train.batch([imu_t,rel_pose_t,imu_len_t],batch_size=batch_size)

	imu_v,rel_pose_v,imu_len_v=read_tfrecords(validation_data)
	imu_v,rel_pose_v,imu_len_v = tf.train.batch([imu_v,rel_pose_v,imu_len_v],batch_size=batch_size)
	
	rel_pose_v = tf.reshape(rel_pose_v,[batch_size, 7])
	Tv12=pose2mat(rel_pose_v)
	Ti12=pose2mat(pred)
	if (do_imu):
		rtd, rta, td, ta =rtLoss(Ti12,Tv12)
		writer_3 = tf.summary.FileWriter('./graphs/e_imu', None)
	if (do_zero_motion):
		T_z=tf.tile(tf.reshape(tf.Variable([[[1,0,0,0],
					 [0,1,0,0],
					 [0,0,1,0],
					 [0,0,0,1]]], dtype=tf.float32),[1,4,4]),[batch_size,1,1])
		rtd, rta, td, ta =rtLoss(T_z,Tv12)	
		writer_3 = tf.summary.FileWriter('./graphs/e_zero', None)
	rtd_e=tf.summary.scalar('rtd', rtd)
	td_e=tf.summary.scalar('td', td)
	rta_e=tf.summary.scalar('rta', rta)
	ta_e=tf.summary.scalar('ta', ta)
	write_op3 = tf.summary.merge([rtd_e, td_e,rta_e,ta_e])
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	with tf.Session()  as sess:	
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		if(is_train):
			step = 1
			# Keep training until reach max iterations
			while step * batch_size < training_iters:
				batch_x, batch_y,imu_len_=sess.run([imu_t,rel_pose_t,imu_len_t])
				batch_y = batch_y.reshape((batch_size, 7))
				# Run optimization op (backprop)
				sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
				#validation on different batch
				batch_x_v, batch_y_v,imu_len_=sess.run([imu_v,rel_pose_v,imu_len_v])
				batch_y_v = batch_y_v.reshape((batch_size, 7))
				if step % display_step == 0:
					# get cost
					val_loss=sess.run(v_cost, feed_dict={x: batch_x_v, y: batch_y_v})
					#loss for train data
					loss = sess.run(t_cost, feed_dict={x: batch_x, y: batch_y})
					print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss)+" Validation loss= ""{:.6f}".format(val_loss))
				# print(step)
				summary1=sess.run(write_op1, feed_dict={x: batch_x, y: batch_y})
				writer_1.add_summary(summary1,step)
				writer_1.flush()
				summary2=sess.run(write_op2, feed_dict={x: batch_x_v, y: batch_y_v})
				writer_2.add_summary(summary2,step)
				writer_2.flush()
				step += 1
			print("Optimization Finished!")
			# Save the variables to disk.	
			save_path = saver.save(sess, "./tmp/model.ckpt")
			print("Model saved in file: %s" % save_path)
		else:
		  # Restore variables from disk.
			saver.restore(sess, "./tmp/model.ckpt")
			print("Model restored.")
			for i in range(1,1000):
				print(i)
				batch_x_v, batch_y_v,imu_len_=sess.run([imu_v,rel_pose_v,imu_len_v])
				batch_y_v = batch_y_v.reshape((batch_size, 7))
				summary3=sess.run(write_op3, feed_dict={x: batch_x_v, y: batch_y_v})
				print(sess.run(Ti12, feed_dict={x: batch_x_v, y: batch_y_v}))
				writer_3.add_summary(summary3,i)
				writer_3.flush()
			print("Done")
	coord.request_stop()
	coord.join(threads)
	# if (using_cluster):
	# 	import os
	# 	os.system('tensorboard --port=1234 --logdir="./graphs/" && exit && cd && ssh -N -4 -L :1234:localhost:1234 compute-0-9;')