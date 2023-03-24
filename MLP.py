from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

class MLP:
    """
    Class implementing Multilayer Perceptron with sigmoid activation functions, which are used
    both in the hidden layers and in the output layer of the model.
    """
    def __init__(self, dims):
        self.dims = dims
        self.weights = {}
        self.L = len(dims)

        for i in range(1, len(dims)):
            self.weights[f'W{i-1}'] = np.random.randn(dims[i], dims[i-1]) * (2/np.sqrt(dims[i-1]))
            self.weights[f'B{i-1}'] = np.zeros((dims[i], 1))
                        
    def derivative(self, A):
        """
        Derivative of the logistic function
        """
        return np.multiply(A, 1 - A)
    
    def activation_function(self, Z):
        """
        Logistic function
        """
        return 1 / (1 + np.exp(-Z))

    def forward(self, X, Y=None):
        """
        Forward pass. We initialize the self.cache dictionary with the vector representing the input layers, denoted A0.
        The forward pass then computes Zi and Ai until reaching the ouput layer. The last matrix A is then returned.
        """
        self.cache = {}
        self.cache['A0'] = X
        
        for i in range(1, self.L):
            Z = np.dot(self.weights[f'W{i-1}'], self.cache[f'A{i-1}']) + self.weights[f'B{i-1}']
            self.cache[f'A{i}'] = self.activation_function(Z)
        return self.cache[f'A{self.L - 1}']
    
    def backward(self, Y):
        """
        This function implements the backward step of the Backprop algorithm. 
        
        The deltas di and gradients dW and dB are stored in self.cache with the keys di, dWi, and dBi, respectively.
        """
        self.cache[f'd{self.L-1}'] = self.cache[f'A{self.L-1}'] - Y
        for i in reversed(range(0,self.L-1)):
            self.cache[f'dW{i}'] = np.dot(self.cache[f'd{i+1}'], np.transpose(self.cache[f'A{i}']))
            self.cache[f'dB{i}'] = np.sum(self.cache[f'd{i+1}'], axis=1, keepdims=True)
            
            if i > 0:
                self.cache[f'd{i}'] = np.multiply(np.dot(np.transpose(self.weights[f'W{i}']), self.cache[f'd{i+1}']), self.derivative(self.cache[f'A{i}']))
            
        
    def update(self, alpha):
        """
        Function uses the dWs and dBs stored in self.cache to update the weights self.weights of the model.
        """
        for i in range(self.L-1):
            self.weights[f'W{i}'] -= alpha / m * self.cache[f'dW{i}']
            self.weights[f'B{i}'] -= alpha / m * self.cache[f'dB{i}']
        
    def train(self, X, Y, X_validation, Y_validation, alpha, steps):        
        # creating one-hot encoding for the labels of the images
        Y_one_hot = np.zeros((10, X.shape[1]))        
        for index, value in enumerate(Y):
            Y_one_hot[value][index] = 1
        
        # performs a number of gradient descent steps
        for i in range(0, steps):
            # computes matrices A and store them in self.cache
            self.forward(X, Y_one_hot)
            
            # computes matrices dW and dB and store them in self.cache
            self.backward(Y_one_hot)
            
            # use the matrices dW and dB to update the weights W and B of the model
            self.update(alpha)
            
            # every 100 training steps we print the accuracies of the model on a set of training and validation data sets
            if i % 100 == 0:
                percentage_train = self.evaluate(X, Y)                
                percentage_validation = self.evaluate(X_validation, Y_validation)
                
                print('Accuracy training set %.3f, Accuracy validation set %.3f ' % (percentage_train, percentage_validation))

    def evaluate(self, X, Y):
        """
        Receives a set of images stacked as column vectors in matrix X, their one-hot labels Y.
        
        Returns the percentage of images that were correctly classified by the model. 
        """
        Y_hat = self.forward(X)
        classified_correctly = test_correct = np.count_nonzero(np.argmax(Y_hat, axis=0) == Y)
        return classified_correctly / X.shape[1]
    
(x_train, y_train), (x_test, y_test) = mnist.load_data()

m = 20000
val = 10000
images, labels = (x_train[0:m].reshape(m, 28*28) / 255, y_train[0:m])
images = images.T

images_validation, labels_validation = (x_train[m:m + val].reshape(val, 28*28) / 255, y_train[m:m + val])
images_validation = images_validation.T

dims = [784, 50, 20, 10]
mlp = MLP(dims)
mlp.train(images, labels, images_validation, labels_validation, 0.5, 3000)