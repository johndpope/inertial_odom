import tensorflow as tf
from io_libs.reader import read_tfrecords
import os

inDir='./records/'
if __name__=="__main__":
	nRecords= len([name for name in os.listdir(inDir) if os.path.isfile(os.path.join(inDir, name))])
	print 'found %d records' % nRecords
	records = [inDir+'/sample_'+ str(i)+'.tfrecord' for i in range(1,nRecords)]
	queue = tf.train.string_input_producer(records, shuffle=False)
 	prev_pose,imu,curr_pose,imu_len=read_tfrecords(queue)
 	BATCH_SIZE=1
	prev_pose,imu,curr_pose,imu_len = tf.train.batch([prev_pose,imu,curr_pose,imu_len],batch_size=BATCH_SIZE)
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	with tf.Session()  as sess:		
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		prev_pose_,imu_,curr_pose_,imu_len_=sess.run([prev_pose,imu,curr_pose,imu_len])
		print(imu_len_)
	coord.request_stop()
	coord.join(threads)