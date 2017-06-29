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
			'rel_pose': tf.FixedLenFeature([], tf.string),
		})
	imu_len=features['imu_len']
	imu = tf.decode_raw(features['imu'], tf.float64)
	rel_pose = tf.decode_raw(features['rel_pose'], tf.float64)
	
	imu = tf.reshape(imu, [20,6],name='imu')
	rel_pose = tf.reshape(rel_pose, [1,7],name='rel_pose')
	
	imu=tf.cast(imu,tf.float32)
	rel_pose=tf.cast(rel_pose,tf.float32)

	return imu,rel_pose,imu_len

if __name__=="__main__":
	nRecords= len([name for name in os.listdir(inDir) if os.path.isfile(os.path.join(inDir, name))])
	print 'found %d records' % nRecords
	records = [inDir+'/sample_'+ str(i)+'.tfrecord' for i in range(1,nRecords)]
	queue = tf.train.string_input_producer(records, shuffle=False)
 	imu,rel_pose,imu_len=read_tfrecords(queue)
 	BATCH_SIZE=10
	imu,rel_pose,imu_len = tf.train.batch([imu,rel_pose,imu_len],batch_size=BATCH_SIZE)
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	with tf.Session()  as sess:		
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		imu_,rel_pose_,imu_len_=sess.run([imu,rel_pose,imu_len])
		print(imu_len_)
	coord.request_stop()
	coord.join(threads)