# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

# Network Parameters
n_steps=20
n_input = 127 # imu (linear acc and ang vel) (20 readings of imu *6 channels) +7 (vicon prev pose)
n_hidden1 = 256 # hidden layer num of features for lstm cell
n_hidden2 = 128 # hidden layer num of features
n_hidden3 = 128 # hidden layer num of features
n_hidden4 = 64 # hidden layer num of features
n_output = 7 # position and quaternion
