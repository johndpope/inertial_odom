import tensorflow as tf
from io_libs.reader import read_tfrecords
import os
from scripts.hyperparams import *
from scripts.lstm import *
import numpy as np

inDir='./records/'
if __name__=="__main__":
	nRecords= len([name for name in os.listdir(inDir) if os.path.isfile(os.path.join(inDir, name))])
	print 'found %d records' % nRecords
	records = [inDir+'/sample_'+ str(i)+'.tfrecord' for i in range(1,nRecords)]
	queue = tf.train.string_input_producer(records, shuffle=False)
	imu,rel_pose,imu_len=read_tfrecords(queue)
	BATCH_SIZE=batch_size
	imu,rel_pose,imu_len = tf.train.batch([imu,rel_pose,imu_len],batch_size=BATCH_SIZE)
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	with tf.Session()  as sess:		
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		# imu_,rel_pose_,imu_len_=sess.run([imu,rel_pose,imu_len])
		# print(np.shape(rel_pose_))
		# print(np.shape(imu_))	
		step = 1
		# Keep training until reach max iterations
		while step * batch_size < training_iters:
			imu_,rel_pose_,imu_len_=sess.run([imu,rel_pose,imu_len])
			# batchx and batchy
			batch_x = imu_.reshape((batch_size, n_steps, n_input))
			batch_y = rel_pose_.reshape((batch_size, 7))
			# Run optimization op (backprop)
			sess.run(x, feed_dict={x: batch_x, y: batch_y})
			if step % display_step == 0:
				# Calculate batch accuracy
				# acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
				# Calculate batch loss
				loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
				print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
				  "{:.6f}".format(loss/batch_size))
			# print(step)
			step += 1
		print("Optimization Finished!")
	    # Calculate accuracy for 128 mnist test images
	    # test_len = 128
	    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
	    # test_label = mnist.test.labels[:test_len]
	    # print("Testing Accuracy:", \
     #    sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
	coord.request_stop()
	coord.join(threads)