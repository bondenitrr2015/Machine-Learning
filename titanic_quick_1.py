# This is the titanic data-set problem
# Current Architecture is 3-5-3-1

import pandas as pd
import tensorflow as tf

# Step 1 - Load and parse the data
# Importing of data
titanic_train = pd.read_csv('/Users/path/to/train.csv')
titanic_test = pd.read_csv('/Users/path/to/test.csv')
titanic_test_result = pd.read_csv('/Users/path/to/genderclassmodel.csv')
# dropping if tables
titanic_train = titanic_train.drop(['PassengerId','Name','Age','Ticket','Cabin','Fare'], axis = 1)
titanic_test = titanic_test.drop(['PassengerId','Name','Age','Ticket','Cabin','Fare'], axis = 1)
# we need to convert gender to 1s and -1s
# my previous method was behaving a bit weirdly so had to make an array and then do things
sex_array = list()
for i in range(len(titanic_train)):
	if titanic_train['Sex'][i] == 'male':
		sex_array.append(1.0)
	else:
		sex_array.append(-1.0)
titanic_train.loc[:, ('sex_usable')] = sex_array

# Step 2 - make label
# Since we alredy have the label in the training set
titanic_survive = pd.DataFrame()
titanic_survive.loc[:, ('Survived')] = titanic_train['Survived'].astype(float)

# Step 3 - prepare the data for tensorflow
# convert features into tensor
'''
inputX = titanic_train.loc[:, ['sex_usable', 'Pclass']].as_matrix()
This input was not giving any good output and so, we cannot use this
'''
inputX = titanic_train.loc[:, ['Pclass', 'Parch', 'sex_usable']].as_matrix()
# convert labels into output tensor
inputY = titanic_survive.loc[:, ['Survived']].as_matrix()
# creating the testing data
inputTest = titanic_test.loc[:, ['Pclass', 'Parch', 'sex_usable']].as_matrix()
outputTest = titanic_test_result.loc[:, ['Survived']].as_matrix()

# Step 4 - write out our hyyperparameters
learning_rate = 0.1
training_epochs = 10000
display_step = 2000
n_samples = inputY.size

# Step 5 - Create our computation graph neural network
n_hidden_1 = 5
n_hidden_2 = 3

x = tf.placeholder(tf.float32, [None, 3])
W1 = tf.Variable(tf.truncated_normal([3,n_hidden_1]))
b1 = tf.Variable(tf.truncated_normal([n_hidden_1]))

W2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2]))
b2 = tf.Variable(tf.truncated_normal([n_hidden_2]))

W3 = tf.Variable(tf.truncated_normal([n_hidden_2, 1]))
b3 = tf.Variable(tf.truncated_normal([1]))

y1 = tf.add(tf.matmul(x, W1), b1)
# y1 = tf.nn.sigmoid(y1)

y2 = tf.add(tf.matmul(y1, W2), b2)
y2 = tf.nn.sigmoid(y2)

y3 = tf.add(tf.matmul(y2, W3), b3)
y3 = tf.nn.sigmoid(y3)

y_ = tf.placeholder(tf.float32, [None, 1])

print("Step five done")

# Step 6 - perform training
cost = tf.reduce_sum(tf.pow(y_ - y3, 2))/(2 * n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# training loop
for i in range(training_epochs):
	sess.run([optimizer], feed_dict = {x: inputX, y_: inputY})

	if i % display_step == 0:
		cc = sess.run(cost, feed_dict = {x: inputX, y_: inputY})
		print("Training Step:", '%04d' % (i), "cost =", "{:.9f}".format(cc))

print("Optimisation Finished")

y = sess.run(y3, feed_dict = {x: inputTest})

def hardLimit(a):
	temp = []
	for i in range(len(a)):
		if a[i] >= 0.5:
			temp.append(1.0)
		else:
			temp.append(0.0)
	return temp

y = hardLimit(y)

corr = 0
total = len(y)
for i in range(total):
	if y[i] == outputTest[i]:
		corr += 1

percentage_correct = (corr / total) * 100

print(percentage_correct, corr, total)

sess.close()