# MLP-LinearRegression

An MLP (Multi-Layer Perceptron) is a type of machine learning system that uses a neural network with multiple layers of interconnected nodes to learn patterns in data. The system takes input data, processes it through the neural network, and produces output predictions or classifications. An MLP is trained using backpropagation, which adjusts the weights and biases of the neurons to minimize the error between the predicted output and the true output for a given set of training examples. This is used to train the neural network on the MNIST handwritten digits dataset.

Linear Regression is a statistical method used to analyze the relationship between two continuous variables, typically referred to as the independent variable and dependent variable. The goal of a linear regression system is to find the linear equation that best fits the data and predicts the value of the dependent variable based on the independent variable. This Linear Regression system uses Gradient Desecent to find the local minimum of the differentiable function.

Similarly, Logistic Regression is used to analyze the relationship between a binary categorical dependent variable and one or more independent variables in a logistic equation and predicts the probability of the depenedent variable taking a particular value based on the values of the independent variable.
* MSE is a loss function used with the goal to predict continuous values. It measures the average squared difference between the predicted values and the true values. The lower the MSE, the better the model fits the data.
* Cross-Entropy Loss on the other hand, is used for classification problems, where the goal is to predict discrete class label. This is clear as our Logistic Regression with CE-Loss consistently outperforms MSE Loss by ~3%. Cross-Entropy Loss measures the difference between the predicted probability distribution and the true probability distribution. The cross-entropy loss is a measure of how well the predicted probabilities match the true probabilities. The lower the cross-entropy loss, the better the model's classification performance.
