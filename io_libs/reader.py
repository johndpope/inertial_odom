import numpy as np
import tensorflow as tf
import os

inDir='../records/'
def read_tfrecords(filename_queue):
	compress = tf.python_io.TFRecordOptions(compression_type = tf.python_io.TFRecordCompressionType.GZIP)
	reader = tf.TFRecordReader(options=compress)
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		features={
			'imu_len': tf.FixedLenFeature([], tf.int64),
			'imu': tf.FixedLenFeature([], tf.string),
			'prev_pose': tf.FixedLenFeature([], tf.string),
			'curr_pose': tf.FixedLenFeature([], tf.string),
		})
	imu_len=features['imu_len']
	imu = tf.decode_raw(features['imu'], tf.float64)
	prev_pose = tf.decode_raw(features['prev_pose'], tf.float64)
	curr_pose = tf.decode_raw(features['curr_pose'], tf.float64)
	
	imu = tf.reshape(imu, [20,6],name='imu')
	prev_pose = tf.reshape(prev_pose, [1,7],name='prev_pose')
	curr_pose = tf.reshape(curr_pose, [1,7],name='curr_pose')
	return prev_pose,imu,curr_pose,imu_len

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