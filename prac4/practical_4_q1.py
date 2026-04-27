####################################################################
# Data Eng 414 - Practical 4 Multilayer Perceptron                 #
# Joshua Jansen van Vuren, Thomas Niesler,                         #
# University of Stellenbosch, 2025                                 #
#                                                                  #
# This skeleton code is to be used in conjunction with the         #
# questions from practical 4 question 1                            #
####################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from helper_functions import read_csv,plot_surface
import random
random.seed(42) # to make experiment repeatable

# Dataset is imported and normalised
# import the dataset
train_X, train_Y = read_csv("data/simple/train.csv")
test_X, test_Y = read_csv("data/simple/test.csv")

# mean and variance normalisation
mean_X = train_X.mean(0)
std_x = train_X.std(0)
train_X = (train_X - mean_X) / std_x
test_X = (test_X - mean_X) / std_x

# ensure the data and target arrays are the intended shapes
print(30 * "*")
print("Data shapes:")
print("Train X: ", train_X.shape)
print("Test X: ", test_X.shape)
print(30 * "*")
print("Train y: ", train_Y.shape)
print("Test y: ", test_Y.shape)
print(30 * "*")

####################################################################
#            (1.a) Import the data and plot the dataset,           #
#       comment on the dataset, is it linerly separable etc.       #
####################################################################
# Map labels to colours for plotting (0 -> red, 1 -> green)
colours = ['r' if label == 0 else 'g' for label in train_Y.flatten()]

plt.figure(figsize=(8, 6))
plt.scatter(train_X[:, 0], train_X[:, 1], c = colours, edgecolors = 'k', alpha = 0.7)
plt.title('Visualization of the Simple Dataset')
plt.xlabel("Normalised Feature 1")
plt.ylabel("Normalised Feature 2")
plt.grid(True, linestyle='--', alpha = 0.6)
plt.savefig('simpledata.png')
print("Plot saved as simpledata.png")

#******************************************************************#

# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# cross-entropy loss for one sample
def loss(y,y_hat):
    eps = 1e-7 # small constant to avoid numerical underflow in log calculation
    return y * np.log(y_hat+eps) + (1-y) * np.log(1-y_hat+eps)

P_1 = 2 # layer 1 has two inputs
Q_1 = 1 # layer 1 has one output 

np.random.seed(42) # to make experiment repeatable
W_1 = np.random.uniform(-1/np.sqrt(Q_1),1/np.sqrt(Q_1),size=(P_1+1,Q_1)) #initialise weights

def forward(u_i,W):
    """
    Arguments:
    ----------
        u[i] : np.ndarray
            - Input vector to layer i (P_i,1) - including bias
        W[i] : np.ndarray
            - Weight matrix of layer i (P_i+1,Q_i)

    Returns:
        z[i] : np.ndarray
            - Linear layer output (Q_i,1)
        v[i] : np.ndarray
            - Activation function output (Q_i,1)
    """
    ####################################################################
    # (1.b) Implement a foward pass for a single layer with a sigmoid  #
    ####################################################################

    z_i = W.T @ u_i  # Linear layer output (Transpose W to match dimensions for matrix multiplication)
    v_i = sigmoid(z_i) # Activation function output

    #******************************************************************#

    return z_i,v_i

####################
# 1.b code checker #
####################
u_i = np.array([[1],[1],[1]])
z_i,v_i = forward(u_i,W_1)
assert np.round(z_i,3) == np.array([[1.114]]) and np.round(v_i,3) == np.array([[0.753]]), "forward pass doesnt pass dummy test"

def loss_derivative(y,y_hat):
    """
    Arguments:
    ----------
    y : np.ndarray
        - Vector of ground truth targets (1,1)
    y_hat : np.ndarray
        - Model predictions (1,1)

    Parameters:
    -----------
    eps : float
        - Small constant to avoid divide by zero error
            Hint: add in the denominator of your calculation.

    Returns:
    --------
    dL_dy_hat : np.ndarray
        - Derivative of loss with respect to model prediction y_hat (1,1)
    """

    eps = 1e-7
    dL_dy_hat = None
    ####################################################################
    #      (1.c) Implement the derivative of binary cross-entropy      #
    #                         from handout 4.4                         #
    ####################################################################

    dL_dy_hat = (y - y_hat) / (y_hat * (1 - y_hat) + eps)  # Add eps to avoid division by zero
    return dL_dy_hat
    #******************************************************************#

####################
# 1.c code checker #
####################
y_hat = np.array([0.5])
y = np.array([0])
ld = loss_derivative(y,y_hat)
assert np.round(ld,3) == np.array([-2.000]), "loss does not pass dummy test"

def backward(u_i,v_i,W_i,Lambda):
    """
    Arguments:
    ---------
        u[i] : np.ndarray
            - Input vector to layer i (P_i,1) - including bias
        v[i] : np.ndarray
            - Activation function output (Q_i,1)
        W[i] : np.ndarray
            - Weight matrix of layer i (P_i,Q_i)

    Returns:
    --------
        dJ_dW[i] : np.ndarray
            - Gradients of weights for layer i
        Lambda : np.ndarray
            - Backpropagation factor for layer i (Slide 6.13) (Q_i,1)
    """

    dJ_dWi = None # How the loss changes with respect to the weights of layer i
    dzi_dvi_prev = None

    ####################################################################
    #       (1.d) Implement the backward pass for a single layer       #
    #                       with a sigmoid activation                  #
    ####################################################################

    delta = Lambda * v_i * (1 - v_i)  # Backpropagation factor for layer i (Q_i,1)
    dJ_dWi = u_i @ delta.T  # Gradients of weights for layer i - we are only dealing with one layer and one perceptron
    Lambda = (W_i[1:, :] @ delta).flatten() # Backpropagation factor for previous layer (Q_i,1) - exclude bias weights in first row of W_i when calculating Lambda for previous layer
    # flattened to 1D array 

    #******************************************************************#

    return dJ_dWi,Lambda

####################
# 1.e code checker #
####################
dJ_dWi,Lambda = backward(u_i,v_i,W_1,ld) # Passed in loss derivative as Lambda (dL_dy_hat)
assert (np.round(dJ_dWi,3) == np.array([-0.372, -0.372, -0.372])).all()
assert (np.round(Lambda,3) == np.array([-0.335, -0.173])).all()

#######################
# set hyperparameters #
#######################
learning_rate = 0.1
epochs = 100
batch_size = 1 # number of samples to average gradients over

train_N = len(train_X) # number of train samples
test_N = len(test_X) # number of test samples

#################
# training loop #
#################

for e in range(epochs):
    train_loss = 0.
    test_loss = 0.

    avg_grads = [np.zeros(W_1.shape)] # initialise list to store gradients for weights (W_1)
    
    idx = list(range(train_N)) # idx contains an ordered list of indexes for each sample in the training set
    random.shuffle(idx) # shuffle that list

    # iterate over a shuffled training set
    for i in idx:
        y = train_Y[i:(i+1),np.newaxis] # extract i'th target from training set
        u_1 = np.hstack([np.ones((1,1)),train_X[i:(i+1),:]]).T # extract i'th input from training set and add bias

        z_1,v_1 = forward(u_1,W_1) # calculate z_i and v_i for this layer

        y_hat = v_1 # there is only one layer so this is output

        train_loss += loss(y,y_hat).item() # accumulate loss over training set

        Lambda = loss_derivative(y,y_hat) # initialise Lambda

        dJ_dw1,Lambda = backward(u_1,v_1,W_1,Lambda) # calculate derivatives by backpropagation

        avg_grads[0] += dJ_dw1 # accumulate gradients for W_1

        # update weights once every 'batch_size' samples 
        if i % batch_size == 0:
            W_1 = W_1 + learning_rate * avg_grads[0] / batch_size # update weights according average gradient over batch_size samples

            avg_grads = [np.zeros(W_1.shape)] # reset list to store gradients for weights (W_1)

    # once per epoch, compute test set loss
    idx = list(range(test_N))
    for i in idx:
        y = test_Y[i:(i+1),np.newaxis] # extract i'th target from training set
        u_1 = np.hstack([np.ones((1,1)),test_X[i:(i+1),:]]).T # extract i'th input from training set and add bias

        z_1,v_1 = forward(u_1,W_1) # calculate z_i and v_i for this layer

        y_hat = v_1 # there is only one layer so this is output

        test_loss += loss(y,y_hat).item() # accumulate loss over test set

    # print the training and test set loss for the epoch
    print(f"Epoch {e:4d} loss: {train_loss/(train_N):.4f} test_loss: {test_loss/(test_N):.4f}")

##################################################################
# (1.f) Use the provided function to plot the prediction surface #
##################################################################

plot_surface([W_1],forward,test_X,test_Y)

# Save result as image
plt.savefig('decision_surface.png')
print("Decision surface plot saved as decision_surface.png")
