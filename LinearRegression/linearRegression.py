#linear regression on the following open source dataset:
#http://archive.ics.uci.edu/ml/datasets/Online+Video+Characteristics+and+Transcoding+Time+Dataset

import csv
from numpy import *
from pylab import *
from scipy import delete

with open('transcoding_mesurment.tsv','r') as input_file:
    csv_data = csv.reader(input_file, delimiter='\t')
    data = list(csv_data)

#transform to numpy array delete columns which are Nan ()
data = delete(data, 0, 0)
data = delete(data, 0, 1)
data = delete(data, 6, 1)
data = delete(data, 6, 1)
data = delete(data, 1, 1)
data = delete(data, 11, 1)

data = data[:1000, :]
data = array(data, dtype=float)
data = data / linalg.norm(data, axis=-1)[:, newaxis]


#plot the first 2 dimentions of the data
#scatter(data[:, 0], data[:, -1])
#title('First 2 dimentions of the data')
#xlabel('X')
#ylabel('Y')
#show()

#Linear regression with one Feature2
X = data
y = data[:, -1]

#number of training examples
m = y.size
n = X[0].size + 1

#make sure they have the right shape
y = y.reshape(m, 1)
print("X shape ", X.shape)
print("y shape ", y.shape)

#Add the intercept (bias) term to X
one_col = ones(shape=(m, 1))
X = concatenate((X, one_col), axis=1)
print("X shape after appending the intercept column", X.shape)

#weights vector
theta = zeros(shape=(n, 1))
print("Theta shape ", theta.shape)

#gradient descent hyperparameters
iterations = 200
alpha = 0.00001

def compute_cost(X, y, theta):
    '''
    Computes the cost of linear regression
    '''
    m = y.size

    predictions = X.dot(theta)
    squared_errors = (predictions - y) ** 2

    J = (1.0 / (2 * m)) * squared_errors.sum()

    return J

def gradient_descent_single_feature(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta)

        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()

        J_history[i, 0] = compute_cost(X, y, theta)
        print(J_history[i, 0])

    return theta, J_history

def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    n = X[0].size
    J_history = zeros(shape=(num_iters, 1))

    #print("Shape of theta, ", theta.shape)

    for i in range(num_iters):

        errors = X.dot(theta) - y

        #print("-------------------------------")
        #print("X has shape ", X.shape)
        #print("Theta has shape ", theta.shape)
        #print("Errors has shape ", errors.shape)

        sigma = []
        for feature in range(0, n):
            fcol = X[:, feature]
            fcol = fcol.reshape(m, 1)

            sigma.append((errors.dot(fcol.T)).sum())

        sigma = array(sigma)
        sigma = sigma.reshape(n, 1)

        theta = theta - alpha * (1 / m) * sigma

        J_history[i, 0] = compute_cost(X, y, theta)
        print(J_history[i, 0])

    return theta, J_history    

#Display initial cost before training
print("Cost before training: ", compute_cost(X, y, theta))

#Perfrom gradient descent on the trainingset
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

#Display the learned weights
print("Learned parameters", theta)    

#Plot the results
#result = X.dot(theta).flatten()
#plot(data[:, 0], result)
#show()

#Plot the cost function
plot(range(0, iterations), J_history)
title("Cost function per iterations")
show()

print("Shape result ", result.flatten().shape)


