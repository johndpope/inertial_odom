import tensorflow as tf
from io_libs.reader import read_tfrecords
import os
from scripts.hyperparams import *
from scripts.lstm import lstm
import numpy as np

inDir='./records/'
if __name__=="__main__":
	nRecords= len([name for name in os.listdir(inDir) if os.path.isfile(os.path.join(inDir, name))])
	print 'found %d records' % nRecords
	records = [inDir+'/sample_'+ str(i)+'.tfrecord' for i in range(1,nRecords)]
	queue = tf.train.string_input_producer(records, shuffle=False)
 	prev_pose,imu,curr_pose,imu_len=read_tfrecords(queue)
 	BATCH_SIZE=batch_size
	prev_pose,imu,curr_pose,imu_len = tf.train.batch([prev_pose,imu,curr_pose,imu_len],batch_size=BATCH_SIZE)
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	with tf.Session()  as sess:		
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		prev_pose_,imu_,curr_pose_,imu_len_=sess.run([prev_pose,imu,curr_pose,imu_len])
		print(np.shape(prev_pose_))
		print(np.shape(imu_))	
		step = 1
	    # Keep training until reach max iterations
		while step * batch_size < training_iters:
			prev_pose_,imu_,curr_pose_,imu_len_=sess.run([prev_pose,imu,curr_pose,imu_len])
			prev_pose_=prev_pose_.reshape((batch_size,7))
			imu_=imu_.reshape((batch_size,120))
		    # batchx and batchy
			batch_x = np.hstack(prev_pose_,imu_)
			batch_y = curr_pose_
		    # Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
			if step % display_step == 0:
			    # Calculate batch accuracy
			    # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
			    # Calculate batch loss
				loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
				print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
		          "{:.6f}".format(loss))
			step += 1
		print("Optimization Finished!")
	coord.request_stop()
	coord.join(threads)