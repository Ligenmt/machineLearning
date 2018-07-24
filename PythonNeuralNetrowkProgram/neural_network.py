import numpy as np
import scipy.special
import matplotlib.pyplot
# %matplotlib inline
import pylab

class neuralNetwork():

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
        self.actication_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.actication_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.actication_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.actication_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.actication_function(final_inputs)
        return final_outputs

# input_nodes = 3
# hiddennodes = 2
# outputnodes = 3
# learning_rate = 0.3
# n = neuralNetwork(input_nodes, hiddennodes, outputnodes, learning_rate)
# n.train([1, 2, 3], [2, 3, 4])



inputnodes = 28 * 28
hiddennodes = 100
outputnodes = 10
learning_rate = 0.2
n = neuralNetwork(inputnodes, hiddennodes, outputnodes, learning_rate)

training_data_file = open('mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(outputnodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

test_data_file = open('mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    output = n.query(np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    label = np.argmax(output) #获取数组中最大值
    print('correct label:', correct_label)
    print('network label:', label)
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = np.asarray(scorecard)
performance = scorecard_array.sum() / scorecard_array.size
print("performance = ", performance)
