# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10 #while training

# Network Parameters
n_steps=20
n_input = 6 # imu (linear acc and ang vel) (20 readings of imu *6 channels) +7 (vicon prev pose)
n_hidden = 256 # hidden layer num of features for lstm cell
n_output = 7 # position and quaternion

# if train or use trained model
is_train=True
if(is_train):
	load_model=False
else:
	load_model=True
# create tf records or read already created ones
create_tfrecords= False
input_Dir='./records/'