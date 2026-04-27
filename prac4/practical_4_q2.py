####################################################################
# Data Eng 414 - Practical 4 Multilayer Perceptron                 #
# Joshua Jansen van Vuren, Thomas Niesler,                         #
# University of Stellenbosch, 2025                                 #
#                                                                  #
# This skeleton code is to be used in conjunction with the         #
# questions from practical 4 question 2                            #
####################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from helper_functions import read_csv,plot_surface
import random
random.seed(42) # to make experiment repeatable

# import the dataset
train_X, train_Y = read_csv("data/two_blob/train.csv")
test_X, test_Y = read_csv("data/two_blob/test.csv")

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
#            (2.a) Import the data and plot the dataset,           #
#       comment on the dataset, is it linerly separable etc.       #
####################################################################

colours = ['r' if label == 0 else 'g' for label in train_Y.flatten()]

plt.figure(figsize=(8, 6))
plt.scatter(train_X[:, 0], train_X[:, 1], c = colours, edgecolors = 'k', alpha = 0.7)
plt.title('Visualization of the Two-Blob Dataset')
plt.xlabel("Normalised Feature 1")
plt.ylabel("Normalised Feature 2")
plt.grid(True, linestyle='--', alpha = 0.6)
plt.savefig('twoblobdata.png')
print("Plot saved as twoblobdata.png")

# Data is NOT linearly separable

#******************************************************************#

# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# cross-entropy loss for one sample
def loss(y,y_hat):
    return y * np.log(y_hat+1e-7) + (1-y) * np.log(1-y_hat+1e-7)


####################################################################
#         (2.b.i) Copy in forward function from question 1         #
####################################################################

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

#******************************************************************#

####################################################################
#          (2.b.ii) Copy in binary cross-entropy loss              #
#                   function from question 1                       #
####################################################################

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

#******************************************************************#

####################################################################
#       (2.b.iii) Copy in backward function from question 1        #
####################################################################

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

    if Lambda.ndim == 1:
        Lambda = Lambda.reshape(-1, 1) 

    dJ_dWi = None # How the loss changes with respect to the weights of layer i
    dzi_dvi_prev = None

    ####################################################################
    #       (1.d) Implement the backward pass for a single layer       #
    #                       with a sigmoid activation                  #
    ####################################################################

    # delta = error signal * derivative of the activation function (sigmoid)
    delta = Lambda * v_i * (1 - v_i)  # Backpropagation factor for layer i (Q_i,1)

    # Gradients of weights for layer i 
    dJ_dWi = u_i @ delta.T  # Gradients of weights for layer i - we are only dealing with one layer and one perceptron

    # Backpropagation factor for previous layer - exclude the bias since is not connected to the previous layer
    Lambda = (W_i[1:, :] @ delta).flatten() # Backpropagation factor for previous layer (Q_i,1) - exclude bias weights in first row of W_i when calculating Lambda for previous layer
    # flattened to 1D array 

    #******************************************************************#

    return dJ_dWi,Lambda

#******************************************************************#

####################################################################
#                 (2.c) Add additional two layers                  #
####################################################################

np.random.seed(42) # to make experiment repeatable

# Adding two additional layers W_2 and W_3 with P_2 = 8 and P_3 = 8 (Q2c)
P_1 = 2 # Input layer has two features
Q_1 = 8 # Number of neurons in first hidden layer = number of outputs

P_2 = 8 # Number of inputs to second hidden layer = number of neurons in first hidden layer
Q_2 = 8 # Number of neurons in second hidden layer = number of outputs

P_3 = 8 # Number of inputs to output layer = number of neurons in second hidden layer
Q_3 = 1 # Final output for binary classification

# Initialise weights for all three layers
W_1 = np.random.uniform(-1/np.sqrt(Q_1),1/np.sqrt(Q_1),size=(P_1+1,Q_1)) # initialise weights for layer W_1
W_2 = np.random.uniform(-1/np.sqrt(Q_2),1/np.sqrt(Q_2),size=(P_2+1,Q_2)) # initialise weights for layer W_2
W_3 = np.random.uniform(-1/np.sqrt(Q_3),1/np.sqrt(Q_3),size=(P_3+1,Q_3)) # initialise weights for layer W_3

#******************************************************************#

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

    avg_grads = [np.zeros(W_1.shape),
                np.zeros(W_2.shape),
                np.zeros(W_3.shape)]  # initialise list to store gradients for weights (W_1,W_2,W_3)

    idx = list(range(train_N)) # idx contains an ordered list of indexes for each sample in the training set
    random.shuffle(idx) # shuffle that list

    # iterate over a shuffled training set
    for i in idx:
        y_hat = None # this should be set in 2.d
        y = train_Y[i:(i+1),np.newaxis] # extract i'th target from training set
        u_1 = np.hstack([np.ones((1,1)),train_X[i:(i+1),:]]).T # extract i'th input from training set and add bias

        z_1,v_1 = forward(u_1,W_1) # calculate z_i and v_i for layer 1

        ###############################################################
        # (2.d) Include the forward passes from the additional layers #
        ###############################################################

        u_2 = np.vstack([np.ones((1, 1)), v_1]) # Add bias to the input of the second layer
        z_2, v_2 = forward(u_2, W_2)  # Calculate z_i and v_i for layer 2

        u_3 = np.vstack([np.ones((1,1)), v_2]) # Add bias to the input of the third layer
        z_3, v_3 = forward(u_3, W_3) # Calculate z_i and v_i for layer 3

        y_hat = v_3 # Final prediction is the activation of the last layer

        #******************************************************************#

        train_loss += loss(y,y_hat).item() # accumulate loss over training set

        Lambda = loss_derivative(y,y_hat) # initialise Lambda

        ###############################################################
        #   (2.e) Include the gradients from the additional layers (backpropagation)   #
        ###############################################################
        # backpropagate the error signal through the layers in reverse order. Propagating the lambda 
        # back through each layer will give us the gradients for the weights of each layer, as well as the lambda for the previous layer to continue backpropagation.

        dJ_dw3, Lambda = backward(u_3, v_3, W_3, Lambda) # Calculate derivatives for layer 3 by backpropagation
        avg_grads[2] += dJ_dw3 # accumulate gradients for W_3

        dJ_dw2, Lambda = backward(u_2, v_2, W_2, Lambda) # Calculate derivatives for layer 2 by backpropagation
        avg_grads[1] += dJ_dw2 # accumulate gradients for W_2

        #******************************************************************#

        dJ_dw1,Lambda = backward(u_1,v_1,W_1,Lambda) # calculate derivatives by backpropagation

        avg_grads[0] += dJ_dw1 # accumulate gradients for W_1
        #************************** YOUR CODE HERE ************************#

        #******************************************************************#

        # update weights once every 'batch_size' samples (in this case batch_size = 1)
        if i % batch_size == 0:
            W_1 = W_1 + learning_rate * avg_grads[0] / batch_size # update weights according average gradient over batch_size samples
            W_2 = W_2 + learning_rate * avg_grads[1] / batch_size
            W_3 = W_3 + learning_rate * avg_grads[2] / batch_size
            
            #******************************************************************#

            avg_grads = [np.zeros(W_1.shape),
                        np.zeros(W_2.shape),
                        np.zeros(W_3.shape)] # reset list to store gradients for weights (W_1,W_2,W_3)

    # once per epoch, compute test set loss
    idx = list(range(test_N))
    for i in idx:
        y_hat = None
        y = test_Y[i:(i+1),np.newaxis] # extract i'th target from training set
        u_1 = np.hstack([np.ones((1,1)),test_X[i:(i+1),:]]).T # extract i'th input from training set and add bias

        # Forward pass through the layers to compute the prediction for the test sample
        z_1,v_1 = forward(u_1,W_1) # calculate z_i and v_i for this layer

        u_2 = np.vstack([np.ones((1, 1)), v_1]) # Add bias to the input of the second layer
        z_2, v_2 = forward(u_2, W_2)

        u_3 = np.vstack([np.ones((1,1)), v_2]) # Add bias to the input of the third layer
        z_3, v_3 = forward(u_3, W_3)

        y_hat = v_3 # Model prediction

        #******************************************************************#

        test_loss += loss(y,y_hat).item() # accumulate loss over test set

    # print the training and test set loss for the epoch
    print(f"Epoch {e:4d} loss: {train_loss/(train_N):.4f} test_loss: {test_loss/(test_N):.4f}")

##################################################################
# (2.f) Use the provided function to plot the prediction surface #
##################################################################

plot_surface([W_1,W_2,W_3],forward,test_X,test_Y)

# Save result as image
plt.savefig('decision_surface_q2.png')
print("Decision surface plot saved as decision_surface_q2.png")
