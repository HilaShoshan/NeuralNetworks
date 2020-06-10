import numpy as np

class Adaline_neuron(object):
    def __init__(self, learningRate=0.01):
        self.learningRate = learningRate
        self.weights = []

    def fit(self, x_train, y_train):
        tempX = x_train.copy()
        tempX[35] = 1
        self.weights = np.array([np.zeros(tempX.shape[1])]).transpose()

        for i in range(tempX.shape[1]):
            xi = tempX.loc[i]
            xi = np.array([xi])
            output = self.activation(xi[0])
            error = y_train.loc[i].values[0] - output[0]
            self.weights += self.learningRate * xi.transpose() * error

    def activation(self, inputs) -> np.ndarray:
        # print(np.dot(inputs, self.weights))
        return np.dot(inputs, self.weights)

    def predict(self, x_test) -> np.ndarray:
        return np.where(self.activation(x_test) >= 0.5, 1, -1)

    def score(self, x_test, y_test):
        output = self.predict(x_test) - y_test
        return len(output[output == 0]) / len(y_test)