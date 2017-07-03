import tensorflow as tf
import numpy as np
import os
import sys
from transforms3d.quaternions import *
import csv

outDir="./records"
inDir="./dataset"
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

def recorder(imu,imu_len,rel_pose):
	global count
	count +=1
	sys.stdout.write("\r processed %d sample" %count)
	sys.stdout.flush()
	recordFile = "%s/sample_%d.tfrecord" % (outDir,count)
	compress = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
	writer = tf.python_io.TFRecordWriter(recordFile, options=compress)
	imu = imu.tostring()
	rel_pose = rel_pose.tostring()
	example = tf.train.Example(features=tf.train.Features(feature={
	  'imu_len': _int64_feature(imu_len),
	  'imu': _bytes_feature(imu),
	  'rel_pose': _bytes_feature(rel_pose),
	}))
	writer.write(example.SerializeToString())
	writer.close()

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
	# print(np.shape(time))
	# print(np.shape(sensor_reading))
	return (time,sensor_reading)
# if __name__ == '__main__':
def write_tfrecords():
	cam_freq=10
	window=int(200/cam_freq) #20
	stride=int(200/100) #2
	n_files=7
	sys.stdout.write("\n Creating TF Records \n --------------------")
	for j in range(1,n_files):
		sys.stdout.write("\n reading vicon room %d \n" %j)
		#200 Hz
		imu_file=inDir+'/v'+str(j)+'/imu0/data.csv'
		(imu_time,imu_data)=read_sensor(imu_file)
		#100 Hz
		vicon_file=inDir+'/v'+str(j)+'/vicon0/data.csv'
		(vicon_time,vicon_data)=read_sensor(vicon_file)
		t_p=0# previous time
		# print(imu_data[0:3,None])
		for i in range(1, len(imu_time)-window,stride):
			# print(np.argmin(abs(vicon_time-imu_time[i,0])), np.min(abs(vicon_time-imu_time[i,0])))
			if (t_p==np.argmin(abs(vicon_time-imu_time[i,0]))):
				continue
			else:
				t_p=np.argmin(abs(vicon_time-imu_time[i,0]))
				prev_pose=vicon_data[np.argmin(abs(vicon_time-imu_time[i,0])),None]
				stack=(imu_data[i:i+window,None])
				curr_pose=vicon_data[np.argmin(abs(vicon_time-imu_time[i+window,0])),None]
				#pose 1
				rot1= quat2mat(prev_pose[0,3:7])
				t1=np.reshape(prev_pose[0,0:3],[1,3])
				T1=np.hstack((rot1,np.transpose(t1)))
				T1=np.vstack((T1,np.array([0,0,0,1])))
				#pose 2
				rot2= quat2mat(curr_pose[0,3:7])
				t2=np.reshape(curr_pose[0,0:3],[1,3])
				T2=np.hstack((rot2,np.transpose(t2)))
				T2=np.vstack((T2,np.array([0,0,0,1])))
				T_rel=np.matmul(np.linalg.inv(T2),T1)
				rel_pose=np.hstack((np.transpose(T_rel[0:3,3]),mat2quat(T_rel[0:3,0:3])))
				recorder(stack,window,rel_pose)	
	print('tf records created in records folder..')
if __name__ == '__main__':
	cam_freq=10
	window=int(200/cam_freq) #20
	stride=int(200/100) #2
	n_files=1
	sys.stdout.write("\n Creating TF Records \n --------------------")
	for j in range(1,n_files):
		sys.stdout.write("\n reading vicon room %d \n" %j)
		#200 Hz
		imu_file=inDir+'/v'+str(j)+'/imu0/data.csv'
		(imu_time,imu_data)=read_sensor(imu_file)
		#100 Hz
		vicon_file=inDir+'/v'+str(j)+'/vicon0/data.csv'
		(vicon_time,vicon_data)=read_sensor(vicon_file)
		print(imu_data[0:3,None])