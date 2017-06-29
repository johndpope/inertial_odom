# Parameters
learning_rate = 0.001
training_iters = 1000000
batch_size = 100
display_step = 100

# Network Parameters
n_steps=20
n_input = 6 # imu (linear acc and ang vel) (20 readings of imu *6 channels) +7 (vicon prev pose)
n_hidden = 256 # hidden layer num of features for lstm cell
n_output = 7 # position and quaternion
