import tensorflow as tf
from io_libs.reader import read_tfrecords
import os
from scripts.lstm import *
import numpy as np
from io_libs.writer import *
from scripts.hyperparams import *

if __name__=="__main__":
	if(create_tfrecords):
		write_tfrecords()
	nRecords= len([name for name in os.listdir(input_Dir) if os.path.isfile(os.path.join(input_Dir, name))])
	print 'found %d records' % nRecords
	records = [input_Dir+'/sample_'+ str(i)+'.tfrecord' for i in range(1,nRecords)]
	queue = tf.train.string_input_producer(records, shuffle=False)
	imu,rel_pose,imu_len=read_tfrecords(queue)
	imu,rel_pose,imu_len = tf.train.batch([imu,rel_pose,imu_len],batch_size=batch_size)
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
				batch_x, batch_y,imu_len_=sess.run([imu,rel_pose,imu_len])
				batch_y = batch_y.reshape((batch_size, 7))
				# Run optimization op (backprop)
				sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
				if step % display_step == 0:
					#loss
					loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
					print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss/batch_size))
				# print(step)
				step += 1
			print("Optimization Finished!")
			# Save the variables to disk.	
			save_path = saver.save(sess, "./tmp/model.ckpt")
			print("Model saved in file: %s" % save_path)
		else:
			  # Restore variables from disk.
			saver.restore(sess, "./tmp/model.ckpt")
			print("Model restored.")
		#tesing on different batch
		batch_x, batch_y,imu_len_=sess.run([imu,rel_pose,imu_len])
		batch_y = batch_y.reshape((batch_size, 7))
		# Run optimization op (backprop)
		print('test',sess.run(cost, feed_dict={x: batch_x, y: batch_y})/batch_size)
		pred_=sess.run(pred,feed_dict={x: batch_x, y: batch_y})
		print('predicted:',pred_[0,:])
		print('original:',batch_y[0,:])
	coord.request_stop()
	coord.join(threads)