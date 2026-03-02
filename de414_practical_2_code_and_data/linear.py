####################################################################
# Data Eng 414 - Practical 2 Linear Regression                     #
# Joshua Jansen van Vuren, Thomas Niesler,                         #
# University of Stellenbosch, 2023-25                              #
#                                                                  #
# This skeleton code is to be used in conjunction with the         #
# questions from practical 2                                       #
####################################################################

import numpy as np

####################################################################
#            (1) Import the data and print out shape               #
####################################################################

# import the data
def read_mnist_csv(filename):
    """
    Reads the MNIST dataset stored in filename:

    Arguments:
    ----------
        filename: str
            - Filename for dataset.
    
    Returns:
        X : np.ndarray
            - Normalised feature vectors
        y : np.ndarray
            - One-hot encoded target class vectors
    """
    X,y = [], []
    with open(filename) as f:
        for line in f:
            # remove end line character and split the CSV data
            clean_line = line.rstrip().split(",")
            
            # add the features to array
            feats = []
            for x in clean_line[1:]:
                feats.append(float(x))
            X.append(feats)

            # add the targets to array
            one_hot = np.zeros(10)
            one_hot[int(clean_line[0])] = 1
            y.append(one_hot)

    # convert to numpy arrays and normalise training features
    X = np.array(X) / 2550
    y = np.array(y)

    return X, y

# import the data from a numpy archive - MUST FASTER!!!
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
            - Values in dictionary are np.ndarray
    """

    data = np.load(filename)

    X = data['X']
    Y = data['Y']

    return X,Y

#************************** YOUR CODE HERE ************************#

#******************************************************************#

####################################################################
#                      The linear regression model                 #
####################################################################

class LinearRegression:
    """Linear regession model capable of batch and iterative
    supervised training.

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

        train_normaleqs : Trains the model utilising the
                    closed-form least-squares regression solution
        
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
        self.K = K

        # initialise weights by sampling from the uniform distribution
        np.random.seed(42) # set random seed for reproducibility
        self.weights = np.random.uniform(-np.sqrt(1/self.D),np.sqrt(1/self.D),(self.D,self.K))


    def forward(self, X):
        """
        Computes the forward pass through the model

        Parameters:
        -----------
            X : np.array of shape (N,D)

        Returns:
        --------
        output np.array of shape (N,K)
        """

        ####################################################################
        #                (2.b) Implement the forward pass                  #
        ####################################################################

        # ensure the input array is the correct shape
        if (len(X.shape) != 2) and X.shape[1] != (self.D - 1):
            raise ValueError("Expected shape of (N,D) where N is" +
            "number of samples and D is: " + str(self.weights.shape[0] - 1) +
            "but recieved shape: " + str(X.shape))
                
        #************************** YOUR CODE HERE ************************#

        #******************************************************************#

    def train_normaleqs(self,X,Y):
        """
        Trains the model utilising the least-squares regression solution,
        the resulting model weights are stored in self.weights

        Parameters:
        -----------
            X : Input np.array of shape (N,D)
            Y : Target np.array of shape (N,K)

        Returns:
        --------
            None
        """

        if (len(X.shape) != 2) and X.shape[1] != (self.D - 1):
            raise ValueError("Expected shape of (N,D) where N is" +
            "number of samples and D is: " + str(self.weights.shape[0] - 1) +
            " but recieved shape: " + str(X.shape))
            
        if (len(Y.shape) != 2) and Y.shape[1] != (self.K):
            raise ValueError("Expected shape of (N,K) where N is" +
            "number of samples but recieved shape: " + str(Y.shape))

        ####################################################################
        #     (2.a) Implement the least-squares regression solution        #
        ####################################################################

        #************************** YOUR CODE HERE ************************#

        #******************************************************************#

    def train(self,X,Y,epochs=200,learning_rate=0.01):
        """
        Trains the model utilising the gradient descent solution.

        Parameters:
        -----------
            X : Input np.array of shape (N,D)
            Y : Target np.array of shape (N,K)
            epochs : Number of times to iterate over the training set
            learning_rate : Learning rate to use in training (alpha)
        """

        ####################################################################
        #             (5) Implement iterative gradient descent             #
        #                 using the batch training protocol                #
        ####################################################################

        if (len(X.shape) != 2) and X.shape[1] != (self.D - 1):
            raise ValueError("Expected shape of (N,D) where N is" +
            "number of samples and D is: " + str(self.weights.shape[0] - 1) +
            " but recieved shape: " + str(X.shape))
            
        if (len(Y.shape) != 2) and Y.shape[1] != (self.K):
            raise ValueError("Expected shape of (N,K) where N is" +
            "number of samples but recieved shape: " + str(Y.shape))

        #************************** YOUR CODE HERE ************************#

        #******************************************************************#

####################################################################
#              (2.c) Create and train the model                    #
####################################################################

#************************** YOUR CODE HERE ************************#

#******************************************************************#

####################################################################
#               (3) Calculate the trained model accuracy           #
#                    and random model performance                  #
####################################################################

def accuracy(Y,Y_hat):
    """
    Calculates the accuracy of model predictions (y_hat) with respect
    to the ground truth labels (Y)

    Arguments:
    ----------
        Y : np.array of shape (N,K)
            - One-hot encoded targets
        Y_hat : np.array of shape (N,K)
            - Digit probabilities/likelihoods for each example in Y
    """

    Y_hat = np.argmax(Y_hat,-1) # get ids of predictions
    Y = np.argmax(Y,-1) # get ids of targets

    acc = np.sum(Y_hat == Y) # count the number of times the ids are the same

    return acc / Y.shape[0] # normalise by the total # targets

#************************** YOUR CODE HERE ************************#

#******************************************************************#

####################################################################
#                (4) Calculate the model error                     #
####################################################################

#************************** YOUR CODE HERE ************************#

#******************************************************************#

####################################################################
#              (5.a) Create and train the model                    #
####################################################################

#************************** YOUR CODE HERE ************************#

#******************************************************************#

####################################################################
#           (5.b) Calculate the trained model error                #
####################################################################

#************************** YOUR CODE HERE ************************#

#******************************************************************#

####################################################################
#                 (5.c) Optional: train model with                 #
#                   alternative hyperparameters                    #
####################################################################

#************************** YOUR CODE HERE ************************#

#******************************************************************#
