import tensorflow as tf
import numpy as np
import os
import sys
outDir="../records"
inDir="../dataset"
count=0
if not os.path.exists(outDir):
	os.makedirs(outDir)

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64s_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def recorder(imu,imu_len,prev_pose,curr_pose):
	global count
	count +=1
	sys.stdout.write("\r processed %d sample" %count)
	sys.stdout.flush()
	recordFile = "%s/sample_%d.tfrecord" % (outDir,count)
	compress = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
	writer = tf.python_io.TFRecordWriter(recordFile, options=compress)
	imu = imu.tostring()
	prev_pose = prev_pose.tostring()
	curr_pose = curr_pose.tostring()
	example = tf.train.Example(features=tf.train.Features(feature={
	  'imu_len': _int64_feature(imu_len),
	  'imu': _bytes_feature(imu),
	  'prev_pose': _bytes_feature(prev_pose),
	  'curr_pose': _bytes_feature(curr_pose),
	}))
	writer.write(example.SerializeToString())
	writer.close()

import csv

def read_sensor(file_name):
	with open(file_name) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		next(readCSV)
		time=np.array([], dtype = np.float64)
		sensor_reading =np.array([], dtype = np.float64)
		for row in readCSV:
			data = np.float64(row[1:None])
			t = np.float64(row[0])
			if (time.size==0):
				time=t
			else:
				time=np.vstack((time,t))
			if (sensor_reading.size==0):
				sensor_reading=data
			else:
				sensor_reading=np.vstack((sensor_reading,data))
	print(np.shape(time))
	print(np.shape(sensor_reading))
	return (time,sensor_reading)

if __name__ == '__main__':
	cam_freq=10
	window=int(200/cam_freq) #20
	stride=int(200/100) #2
	#200 Hz
	imu_file=inDir+'/v'+str(1)+'/imu0/data.csv'
	(imu_time,imu_data)=read_sensor(imu_file)
	#100 Hz
	vicon_file=inDir+'/v'+str(1)+'/vicon0/data.csv'
	(vicon_time,vicon_data)=read_sensor(vicon_file)
	t_p=0# previous time
	for i in range(1, len(imu_time)-window,stride):
		# print(np.argmin(abs(vicon_time-imu_time[i,0])), np.min(abs(vicon_time-imu_time[i,0])))
		if (t_p==np.argmin(abs(vicon_time-imu_time[i,0]))):
			continue
		else:
			t_p=np.argmin(abs(vicon_time-imu_time[i,0]))
			prev_pose=vicon_data[np.argmin(abs(vicon_time-imu_time[i,0])),None]
			stack=(imu_data[i:i+window,None])
			curr_pose=vicon_data[np.argmin(abs(vicon_time-imu_time[i+window,0])),None]
			recorder(stack,window,prev_pose,curr_pose)	