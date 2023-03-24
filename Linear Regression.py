from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# using 20k out of the 60k images in the set
training_size = 20000

# flattening to images to 28x28 to a vector of size 784
images, labels = (x_train[0:training_size].reshape(training_size, 28*28) / 255, y_train[0:training_size])
images = images.T

# Here we apply the same transformations described above on the test data. 
images_test = x_test.reshape(x_test.shape[0], 28*28) / 255
images_test = images_test.T

class LinearRegression:
    
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Constructor assumes a x_train matrix in which each column contains an instance.
        Vector y_train contains one integer for each instance, indicating the instance's label. 
        
        Constructor initializes the weights W and B, alpha, and a one-vector Y containing the labels
        of the training set. Here we assume there are 10 labels in the dataset. 
        """
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._m = x_train.shape[1]
        
        self._W = np.random.randn(10, 784) * 0.01
        self._B = np.zeros((10, 1))
        self._Y = np.zeros((10, self._m))
        self._alpha = 0.05

        for index, value in enumerate(labels):
            self._Y[value][index] = 1
            
    def return_weights_of_digit(self, digit):
        """
        Returns the weights of the model for a given digit
        """
        return self._W[digit, :]
            
    def h_theta(self, X):
        """
        Computes the value of the hypothesis according to the linear regression rule
        """
        
        return np.dot(self._W, X) + self._B
    
    def train(self, iterations):
        """
        Performs a number of iterations of gradient descend equals to the parameter passed as input.
        
        Returns a list with the percentage of instances classified correctly in the training and in the test sets.
        """
        classified_correctly_train_list = []
        classified_correctly_test_list = []
        
        for i in range(iterations):
            # compute the value produced by the model (A)
            # The pure error for all training instances (pure_error)
            # And adjust the matrices self._W and self._B according to the gradient descent rule
            A = self.h_theta(self._x_train)
            pure_error = A - self._Y
            self._W -= self._alpha * np.dot(pure_error, self._x_train.T) / self._m
            self._B -= self._alpha * np.sum(pure_error, axis=1, keepdims=True) / self._m

            if i % 100 == 0:
                classified_correctly = np.sum(np.argmax(A, axis=0) == np.argmax(self._Y, axis=0))
                percentage_classified_correctly = (classified_correctly / self._m) * 100
                classified_correctly_train_list.append(percentage_classified_correctly)
                
                Y_hat_test = self.h_theta(images_test)
                test_correct = np.count_nonzero(np.argmax(Y_hat_test, axis=0) == self._y_test)
                classified_correctly_test_list.append((test_correct)/len(self._y_test) * 100)
                
                print('Accuracy train data: %.2f' % percentage_classified_correctly)
        return classified_correctly_train_list, classified_correctly_test_list

lin_reg = LinearRegression(images, labels, images_test, y_test)
print('Training Linear Regression')
percentage_lin_reg_train, percentage_lin_reg_test = lin_reg.train(3000)