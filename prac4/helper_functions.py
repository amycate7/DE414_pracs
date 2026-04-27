import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# import the data
def read_csv(filename):
    """
    Reads the BLOB dataset stored in filename:

    Arguments:
    ----------
        filename: str
            - Filename for dataset.
    
    Returns:
    --------
        X : np.ndarray
            - Normalised feature vectors
        y : np.ndarray
            - One-hot encoded target class vectors
    """
    X,y = [], []
    with open(filename) as f:
        f.readline() # read header line

        for line in f:
            # remove end line character and split the CSV data
            clean_line = line.rstrip().split(",")
            
            # add the features to array
            feats = []
            for x in clean_line[1:]:
                feats.append(float(x))
            X.append(feats)

            # add the targets to array
            y.append(int(float(clean_line[0])))

    # convert to numpy arrays and normalise training features
    X = np.array(X)
    y = np.array(y)

    return X, y

def plot_surface(W,forward,val_X,val_Y,mode="show"):
    """
    Plots of 2D prediction surface over X={-2,2} and Y={-2,2}
    given weights for a MLP network.

    Arguments:
    ----------
        W : list[np.ndarray]
            - List of model weights 
        forward : function
            - Forward pass function for the model
        val_X : np.ndarray
            - Validation set X (features) values for a scatter plot
        val_Y : np.ndarray
            - Validation set Y (target) values for a scatter plot
    
    Returns:
    --------
        None
    """

    X_1 = np.arange(-2, 2, 0.04)
    X_2 = np.arange(-2, 2, 0.04)
    X_1_orig, X_2_orig = np.meshgrid(X_1, X_2)

    X_1 = np.reshape(X_1_orig,(100*100))
    X_2 = np.reshape(X_2_orig,(100*100))

    X = np.vstack([X_1,X_2]).T

    Y_pred = []

    for x in X:
        x =  np.hstack([np.ones((1,1)),x[np.newaxis,...]]).T

        for w in W:
            _,v_i = forward(x,w)

            x = np.vstack([np.ones((1,1)),v_i])

        Y_pred.append(v_i)

    Y_pred = np.reshape(np.hstack(Y_pred),(100,100))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X_1_orig, X_2_orig, Y_pred, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True,zorder=0,alpha=0.5)

    ax.scatter(val_X[:,0],val_X[:,1],(val_Y*1.05 - 0.025),c=['r' if c == 1 else 'b' for c in val_Y],zorder=100,alpha=1)

    if mode == "show":
        plt.show()
    elif mode == "save":
        plt.savefig("surface.png")