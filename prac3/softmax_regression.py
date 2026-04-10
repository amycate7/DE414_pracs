####################################################################
# Data Eng 344 - Practical 2 Linear Regression                     #
# Joshua Jansen van Vuren, Thomas Niesler,                         #
# University of Stellenbosch, 2022-2025                                 #
#                                                                  #
# This skeleton code is to be used in conjunction with the         #
# questions from practical 3                                       #
####################################################################

import os
import numpy as np
from math import ceil
import matplotlib.pyplot as plt

# TO-DO: q4 (logging/check question answered properly) + q5 (hyperparameter search)
# Write report too ;)

####################################################################
#                Method to import the data                         #
####################################################################

# import the data

def read_mnist_npz(filename):
    """
    Reads the MNISTS dataset from a numpy archive.
    
    Arguments:
    ----------
        filename: str
            - Filename for numpy archive

    Returns:
    --------
        X : np.ndarray
            - Normalised feature vectors
        Y : np.ndarray
            - One-hot encoded target class vectors

    Notes:
    ------
        .npz files are compressed binary formats created using:
            - np.savez_compressed("filename.npz",arr1=arr1,arr2=arr2,...)
        Which can be loaded by:
            - data = np.load(filename)
        Returing a dict with keys -> arr1,arr2,....
            - arr1 = data['arr1']
    """

    data = np.load(filename)

    X = data['X']
    Y = data['Y']

    return X,Y


####################################################################
#          (Q1) Write a function to compute the softmax            #
####################################################################

def softmax(X,axis=-1):
    """
    Calculates the softmax activation of an N-dimensional
    vector X.

    To ensure numerical stability, the maximum value of the input vector 
    is subtracted from every element before exponentiation. 
    Prevent over/underflow errors.

    Vectorized implementation (significantly faster than loops for large data matrices)

    Parameters:
    -----------
        X : np.ndarray
            - Input array
        axis : int
            - Indicates the axis which should be normalised.
              Defaults to the last dimension (-1).

    Returns:
    --------
        Vector after softmax activation 
    """

    # Subtract the maxmimum for numerical stability
    shift_X = X - np.max(X, axis = axis, keepdims = True)

    # Compute the exponentials
    exps = np.exp(shift_X)

    # Compute softmax
    return exps / np.sum(exps, axis = axis, keepdims = True)

####################################################################
#     (Q2) Define a function to compute the log-likelihood        #
####################################################################

def ll(Y,Y_hat):
    """
    Calculates the log likelihood between one-hot encoded
    target vectors (Y), and predictions (Y_hat).

    Parameters:
    -----------
        Y : np.ndarray
            - One hot ground truth (N,K)
        Y_hat : np.array
            - Probabilities (N,K)

    Returns:
    --------
        LL : np.ndarray
            - Log likelihoods per sample (N)
    """
    # Perturb Y_hat to prevent log(0) which is undefined
    epsilon = 1e-15
    Y_hat = np.clip(Y_hat, epsilon, 1)

    log_probs = np.log(Y_hat) # Compute the log of the predicted probabilities
    LL = np.sum(Y * log_probs, axis = 1) # Element-wise multiplication of Y and log_probs, then sum across classes for each sample

    return LL # Log likelihoods per sample

    ####################################################################
    # (Q3) Implement a function to extract a minibatch from a dataset  #
    ####################################################################

def get_mini_batch(X, Y, i, batch_size = 60):
    """
    Returns a minibatch of specified size from a specified index in the provided dataset
    The default size is 60.

    Slice data matrix X and vector Y from index i to i + batch_size

    Parameters:
    -----------
        X : np.ndarray
             - Matrix of feature vectors, shape (N,D)
        Y : np.array
            - Matrix of one-hot encoded target vectors, shape (N,K)
        i : index of batch to be extracted, scalar (batch number)
        batch_size: number of feature vectors in the minibatch, scalar

    Returns:
    --------
        X_mini : np.ndarray
            - Minibatch of feature vectors, shape (size,D)
        Y_mini : np.ndarray
            - Minibatch of one-hot encoded target vectors, shape (size,K)
    """

    start = i * batch_size # Starting index of the batch
    end = start + batch_size # Ending index of the batch

    # Slice the data matrix X and vector Y to get the minibatch
    X_mini = X[start:end] # Slicing along first axis (rows)
    Y_mini = Y[start:end]

    return X_mini, Y_mini

def accuracy(Y,Y_hat):
    """
    Calculates the accuracy between one-hot encoded
    target vectors (Y), and predictions (Y_hat).

    Parameters:
    -----------
        Y : np.ndarray
            - One hot ground truth (N,K)
        Y_hat : np.array
            - Probabilities (N,K)

    Returns:
    --------
        Percentage number of correct predictions 
    """

    hits = np.equal(np.argmax(Y_hat,-1),np.argmax(Y,-1)).astype("int")

    return np.sum(hits) / Y_hat.shape[0]


####################################################################
#                  (Q4) The logistic regression model              #
####################################################################

class MultinomialLogisticRegression:
    """Multinomial Logistic regession model

    Attributes:
    -----------
    self.weights : numpy.ndarray of shape (D,K) where D
        is dimensionality of of weight vector (including
        bias) and K is the number of classes

    self.D : int
        - dimesionality of model (including bias)

    self.K : int
        - dimensionality of targets

    Methods:
    --------
        forward : Computes the forward pass
        
        train : Trains the model utilising gradient
                    descent
    """

    def __init__(self,D,K):
        """
        Constructs the weight matrix

        Parameters:
        -----------
            D : int
                - dimesionality of model (not including bias)
            K : int
                - dimensionality of targets

        Variables:
        ----------
            self.D : int
                - dimesionality of model (including bias)

            self.K : int
                - dimensionality of targets
            
            self.weights : numpy.ndarray of shape (D,K) where D
                is dimensionality of of weight vector (including
                bias) and K is the number of classes
        """

        self.D = D + 1 # add 1 to size for bias
        self.K = K # Number of classes

        # initialise weights by sampling from the normal distribution
        np.random.seed(73) # set random seed for reproducibility
        self.weights = np.random.normal(0,1e-2,(self.D,self.K))

    def forward(self, X):
        """
        Computes the forward pass through the model

        Parameters:
        -----------
            X : np.array of shape (N,D)

        Returns:
        --------
            Y_hat : np.array of shape (N,K)
        """

        ####################################################################
        #               (Q4.a) Implement the forward pass                  #
        ####################################################################

        # ensure the input array is the correct shape
        if (len(X.shape) != 2) and X.shape[1] != (self.D - 1):
            raise ValueError("Expected shape of (N,D) where N is" +
            "number of samples and D is: " + str(self.weights.shape[0] - 1) +
            "but recieved shape: " + str(X.shape))
        
        # Add bias term to input data (column of 1s to X matrix)
        ones = np.ones((X.shape[0], 1)) # Create a column of ones with the same number of rows as X
        X_bias = np.hstack([ones, X]) # Append the column of ones to the left of X
        
        logits = X_bias @ self.weights
        Y_hat = softmax(logits, axis = 1)
        return Y_hat
        

    def train(self,train_X,train_Y,test_X,test_Y,epochs=32,learning_rate=0.01,batch_size=60):
        """
        Trains the model utilising minibatch gradient descent, trained weights
        are stored in self.weights

        Parameters:
        -----------
            train_X : np.array
                - Training input features of shape (N,D)
            train_Y : np.array
                - Training set one-hot encoded target vectors of shape (N,K)
            test_X : 
                - Test input features of shape (N,D)
            test_Y : 
                - Test set one-hot encoded target vectors of shape (N,K)
            epochs : int 
                - Number of times to iterate over the training set
                - epoch := one complete pass over the entire training set
            learning_rate : float 
                - Learning rate to use in training (alpha)
            batch_size : int
                - Minibatch size to average gradients over
        """

        ####################################################################
        #     (Q4.b) Implement iterative gradient descent using the        #
        #            minibatch training protocol                           #
        #                                                                  #
        #     (Q4.d) Add functionality to log the training and test set    #
        #            loss and accuracy to "log/log.csv"                    #
        ####################################################################

        if (len(train_X.shape) != 2) and train_X.shape[1] != (self.D - 1):
            raise ValueError("Expected shape of (N,D) where N is" +
            "number of samples and D is: " + str(self.weights.shape[0] - 1) +
            " but recieved shape: " + str(train_X.shape))
            
        if (len(train_Y.shape) != 2) and train_Y.shape[1] != (self.K):
            raise ValueError("Expected shape of (N,K) where N is" +
            "number of samples but recieved shape: " + str(train_Y.shape))
        
        log_file = "log/log.csv"
        os.makedirs("log", exist_ok=True) # Create log directory if it doesn't exist

        with open(log_file,"w") as f:
            f.write("epoch,train_ll,test_ll,train_acc,test_acc\n")

        num_samples = train_X.shape[0]
        num_batches = num_samples // batch_size 

        for epoch in range(epochs):
            for i in range(num_batches):
                x_batch, y_batch = get_mini_batch(train_X, train_Y, i, batch_size)

                y_hat = self.forward(x_batch) # Forward pass to get predictions for the batch

                ones = np.ones((x_batch.shape[0], 1)) # Create a column of ones with the same number of rows as x_batch
                x_batch_bias = np.hstack([ones, x_batch]) # Append the column of ones to the left of x_batch

                # Calculate the gradient of the log-likelihood wrt. to the weights
                grad = (x_batch_bias.T @ (y_batch - y_hat)) / batch_size # Average over the batch

                self.weights += learning_rate * grad
            
            # Log training and test set log-likelihood and accuracy after each epoch
            train_Y_hat = self.forward(train_X)
            test_Y_hat = self.forward(test_X)
            train_ll = np.mean(ll(train_Y, train_Y_hat))
            test_ll = np.mean(ll(test_Y, test_Y_hat))
            train_acc = accuracy(train_Y, train_Y_hat)
            test_acc = accuracy(test_Y, test_Y_hat)
            with open(log_file, "a") as f:
                f.write(f"{epoch+1},{train_ll},{test_ll},{train_acc},{test_acc}\n")


####################################################################
#               Import training and test data                      #
####################################################################

# Training set
train_X, train_Y = read_mnist_npz("data/train.npz")
print("Training set read.")
print("   train_X is",train_X.ndim,"dimensional",train_X.shape,"so that N =",train_X.shape[0],"and each input has",train_X.shape[1],"dimensions.") 
print("   train_Y is",train_Y.ndim,"dimensional",train_Y.shape,"so that K =",train_Y.shape[1]," classes.") 
 
 # Test set
test_X, test_Y = read_mnist_npz("data/test.npz")
print("Test set read.")
print("   test_X is",test_X.ndim,"dimensional",test_X.shape,"so that N =",test_X.shape[0],"and each input has",train_X.shape[1],"dimensions.") 
print("   test_Y is",test_Y.ndim,"dimensional",test_Y.shape,"so that K =",test_Y.shape[1]," classes.") 
 

####################################################################
#             (Q4.c) Create and train the model                    #
####################################################################

lr = MultinomialLogisticRegression(train_X.shape[1], train_Y.shape[1])
lr.train(train_X, train_Y, test_X, test_Y, epochs=32, learning_rate=0.01, batch_size=60)
y_hat = lr.forward(test_X)
print("Test set accuracy: ", accuracy(test_Y, y_hat))

#******************************************************************#

####################################################################
#   (Q5) Perform a hyperparameter search for the optimal batch     #
#        size and number of training epochs                        #
####################################################################

#************************* YOUR CODE HERE *************************#

#******************************************************************#


####################################################################
print("===== END OF CODE =====")
####################################################################
