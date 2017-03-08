# This is a simple tensorflow code to run a XOR neural network

import tensorflow as tf
import numpy as np

# Step 1 : Loading of data
xor_train = np.array([[0,0], [0,1], [1,0], [1,1]]) #[input_1, input_2]

# Step 2 : Make labels
xor_result_1 = np.array([[0], [1], [1], [0]])

# Step 3 : prepare data for tensorflow
inputX = xor_train
inputY = xor_result_1

# Step 4 : Defining the hyper-parameters
learning_rate = 0.1
training_epochs = 20000
n_samples = inputY.size

# Step 5 : Creating the computation graph
x = tf.placeholder(tf.float32, shape = [4, 2])

W1 = tf.Variable(tf.truncated_normal([2,2], 0.1))
b1 = tf.Variable(tf.truncated_normal([2], 0.1))

W2 = tf.Variable(tf.truncated_normal([2,1], 0.1))
b2 = tf.Variable(tf.truncated_normal([1], 0.1))

y1_values = tf.add(tf.matmul(x, W1), b1)
y1_values = tf.nn.sigmoid(y1_values)

y2_values = tf.add(tf.matmul(y1_values, W2), b2)
y = tf.nn.sigmoid(y2_values)

y_ = tf.placeholder(tf.float32, shape = [4, 1])

# Step 6 : Training the neural network
cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2 * n_samples)
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# opt_w = opt.minimize(cost, var_list = [W, b])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# training loop
for i in range(training_epochs):
	sess.run([opt], feed_dict = {x: inputX, y_: inputY})

print("y =",sess.run(y, feed_dict = {x: inputX, y_: inputY}))