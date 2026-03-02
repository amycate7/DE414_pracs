####################################################################
# Data Eng 414 - Python Practice                                   #
# Joshua Jansen van Vuren, Thomas Niesler,                         #
# University of Stellenbosch, 2023                                 #
#                                                                  #
# This skeleton aims to help get familiar with python, loading     #
# data into memory, making predictions with models, and            #
# calculating accuracy                                             #
####################################################################

####################################################################
#               (1) Import numpy and matplotlib                    #
####################################################################

import numpy as np
import matplotlib.pyplot as plt

#******************************************************************#

####################################################################
#   (2) Read in the MNIST dataset from files data/train.csv and    #
#       data/test.csv and store as numpy variables with shapes:    #                                         
#                   train : (60000,784)                            #
#                   test : (10000,784)                             #
#                   train targets : (60000,1)                      #
#                   test targets : (10000,1)                       #
#                                                                  #
#    Note: * open data/train.csv to see how the data is stored:    #
#          target,pixel_0_0,pixel_0_1,...,pixel_28_27,pixel_28_28  #
#                                                                  #
#          * The MNIST data set consists of 28 x 28 pixel          #
#          handwritten digits where each pixel has a value         #
#          between 0 and 255                                       #
####################################################################


# Define an empty list to store the training data
data = []

# Read the data line by line into the list
with open("data/train.csv") as f:
    for line in f:
        data.append(line.rstrip().split(","))

# Convert the list into a numpy array
data = np.array(data, dtype = np.int32)

# Slice data to seperate training from target data
train = data[:,1:]
train_target = data[:,0]

# Repeat this process for the test data
test_data = []

with open("data/test.csv") as f:
    for line in f:
        test_data.append(line.rstrip().split(","))

test_data = np.array(test_data, dtype = np.int32)

# Slice the test data into raw data and its corresponding targets
test = test_data[:,1:785]
test_target = test_data[:,0]


#******************************************************************#

####################################################################
#       (3) Print the shapes of the training and test sets         #
####################################################################

print(train.shape) # Prints the dimensions of the training data to the terminal
print(train_target.shape)
print(test.shape)
print(test_target.shape)

#******************************************************************#

####################################################################
#           (4) Normalise the dataset to the range 0,1             #
####################################################################

# Normalise data
# The MNIST pixels are integers in the range [0, 255]. Dividing by 255.0
# scales the values to the range [0.0, 1.0], which is helpful for numerical
# stability when training or using machine-learning models.
train_norm = train / 255.0
test_norm = test / 255.0

# Quick sanity checks to confirm the normalization worked as expected
min_train = np.min(train_norm)
max_train = np.max(train_norm)
min_test = np.min(test_norm)
max_test= np.max(test_norm)

print(f"The min of the training data is {min_train} and the maximum is {max_train}")
print(f"The min of the test data is {min_test} and the maximum is {max_test}")

#******************************************************************#

####################################################################
#            (5) Plot an example from the loaded datset            #
####################################################################

train_ex = train_norm[0,:] # Select the data from the first row
train_ex = np.reshape(train_ex, (28, 28)) # Reshape the pixel data to its dimensions so that it can be correctly plotted

plt.imshow(train_ex) # Plot the data
plt.show()

#******************************************************************#

####################################################################
#      (6) Load in weights from a logistic regression model from   #
#           the file data/weights.txt                              #
#                                                                  #
#          * Store the weights as a numpy array called weights     #
#            the shape of the array should be (785, 10)            #
####################################################################

weights = [] # Create an empty list

# Read in the weights line by line into the list
with open("data/weights.csv", 'r') as f:
    for line in f:
        weights.append(line.rstrip().split(","))

# Read the weights into a numpy array
weights = np.array(weights, dtype = np.float32)
np.reshape(weights, (785, 10))   

#******************************************************************#

####################################################################
#  (7) Use the loaded weights to create a model by uncommenting    #
#      the following lines, then use the model to make a           #
#      prediction on an example from the test set                  #
#      * For an interesting result look at index 7                 #
#                                                                  #
#       Note: When feeding input to the model make sure the shape  #
#             of the input is a row vector (1,784)                 #    
#              i.e. taking a row from the X matrix whose shape     #
#                   is (N,D)                                       #
#                                                                  #
#                                                                  #
#       * Load the model weights by using the class function       #
#       model.load() - for more information call help(model.load)  #
####################################################################

#******************************************************************#

from models import SoftmaxRegression
model = SoftmaxRegression()

# Returns a (1,10) array of probabilities. This row vector of length 10 contains the predicted probabilities for each digit 
model.load(weights)

# Extract an example from test set
example = test_norm[7,:]
example = example.reshape((784,1)).T

prob = model(example)
max_index = np.argmax(prob) #To obtain the predicted class id use `np.argmax(prob)` which returns the index of the maximum probability.

print(f"The predicted value of the mnist digit example is {max_index}") # Print out the predicted value.

#******************************************************************#


####################################################################
#   (8) Plot a bar graph of the model probabilities                #
#                                                                  #
#       * Also plot which number you are trying predict            #
####################################################################

x_axis = np.arange(0, 10) # Create arrays of evenly spaced values between 0 and 10
plt.bar(range(10), prob[0]) # Pass in x-axis positions and corresponding values
plt.show()

#******************************************************************#

####################################################################
#        (9) Define a function to calculate accuracy               #
#                                                                  #
#       * Then calculate the average for the test set              #
#                                                                  #
#     * Note: You can retrieve the model prediction by finding     #
#       the index whose probability is the maximum in the array    #
#      this can be accomplished using numpys argmax() function     #
#          prediction_id = np.argmax([0,0.1,0.2,0.5,0.2])          #
#          Results in prediction_id = 3                            #
#                                                                  #
#                                                                  #
#     * Additional note: If you have two arrays with equal shapes  #
#         you can find the elements are equal by using numpys      #
#               equal function.                                    #
#          tar = [1,2,3,4]                                         #
#          pred = [4,3,3,4]                                        #
#          eq = np.equal(pred,tar)                                 #
#           Results in:                                            #
#           eq = [False,False,True,True]                           #
#                                                                  #
#              Hint: np.sum(eq) is a quick way to count the        #
#                     number of correct predictions                #
#                                                                  #
####################################################################

# Function to determine accuracy
def accuracy(num_correct, total_pred):
    return num_correct/total_pred # Divide the number of total correct classifications by total predictions

test_pred = [] # Define an empty list to store the test data classification results

# Loop through every data instance (row)
for row in test_norm:
    row = row.reshape((784,1)).T
    probs = model(row) # Returns an array with 10 indices, storing the probability of the instance being classified as digit i
    max_index = np.argmax(probs) # Classification result determined using index of array element containing the maximum probability

    # Append new row of probabilities for next digit
    test_pred.append(max_index)

test_pred = np.array(test_pred) # Convert the list to a numpy array

eq = np.equal(test_pred, test_target) # Test whether our model classifications are equal to their corresponding target values
total_correct = np.sum(eq) # Sum the total correct classifications

acc = accuracy(total_correct, test_target.size) # Use the function to determine accuracy of classifier
print(acc) # Print accuracy to the terminal

#******************************************************************#