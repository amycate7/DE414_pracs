import numpy as np

class SoftmaxRegression:
    def __init__(self,feature_size = None,n_classes = None) -> None:
        np.random.seed = 42
        # we increase feature size to include bias variable
        # and handle the incorporation of the extra feature into X automatically
        if feature_size == None:
            self.w = 0
            self.feature_size = 0
        else:
            self.w = np.random.normal(loc=0,scale=0.01,size=(feature_size+1,n_classes))
            self.feature_size = feature_size+1
        self.threshold = 0.5
        self.n_classes = n_classes

    def __call__(self, X, **kwds):
        X_ = np.concatenate((np.ones((X.shape[0],1)),X),axis=-1)
        return self.softmax(X_ @ self.w)

    def grad(self,X,y):

        y_hat = self(X)

        X_ = np.concatenate((np.ones((X.shape[0],1)),X),axis=-1)

        return np.dot(X_.T,(y-y_hat))

    def predict(self,X):

        y_hat = self(X)
        y_hat_argmax = np.expand_dims(np.argmax(y_hat,-1),-1)
        enum = np.expand_dims(np.arange(y_hat_argmax.shape[0]),-1)
        y_hat_argmax = np.concatenate((enum,y_hat_argmax),-1)

        pred = np.zeros((y_hat.shape))
        pred[y_hat_argmax[:,0],y_hat_argmax[:,1]] = 1

        return pred

    def nll(self,X,y):
        y_hat = self(X) # (n,10)

        ll = y * np.log(y_hat)

        return -np.sum(ll,-1)
    
    def load(self,weights):
        """Loads a model with trained weights

        Parameters:
        -----------
           weights: Numpy array of shape (D,K)
                D - dimesionality of model
                K - dimensionality of targets
                Note that D includes bias
        """
        self.w = weights

        self.feature_size = weights.shape[0]

    @staticmethod
    def softmax(inp):
        # assume input has shape (samples,classes)
        return np.exp(inp) / np.expand_dims(np.sum(np.exp(inp),-1),-1)