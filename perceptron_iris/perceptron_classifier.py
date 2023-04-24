import numpy as np
from iris_dataset import df as iris_df
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron_Classifier(object):
    '''
    Parameters
    -------------
    learning_rate : float
        Learning rate (between 0.0 and 1.0). Decides how much the weights are updated in each iter. (typically, small).
    n_iter : int
        Number of iterations over the training dataset. 
    random_seed : int
        Random number generator seed for random weight initialization (
    Attributes
    -------------
    These attributes are used to keep track of the Perceptron object.

    w_ : 1d_array
        Weights after fitting. Stores the weights learned during training
    errors_ : list
        Number of misclassifications (updates) in each epoch (training iteration)
    '''
    def __init__(self, learning_rate=0.01, n_iter=50, random_seed=1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_seed = random_seed
    
    def fit(self, X, y):
        """
        Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors. 2-dimensional array where n_examples is the 
            number of training examples and n_features is the number of features for each example
        y : array-like, shape = [n_examples]
            Target values. 1-dimensional array of shape [n_examples] containing the corresponding target output values for each example.

        ***note: Capitalisation of X and y is done to distinguish between input features and output targets. Common in ML notation.
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_seed)
        self.w_ = rgen.normal (loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        '''
        Calculate the net input of the perceptron.

        Parameters:
        -----------
        X : {array-like}, shape = [n_examples, n_features]
            Input data.

        Returns:
        --------
        net_input : numpy array, shape = [n_examples]
            Net input calculated as the dot product of the input data and the 
            weight coefficients of the perceptron, plus the bias unit.
        '''        
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def predict(self, X):
        '''
        Return the predicted class labels after applying the Heaviside (classfy as either 1 or -1) step function.

        Parameters:
        -----------
        X : {array-like}, shape = [n_examples, n_features]
            Input data.

        Returns:
        --------
        y_pred : numpy array, shape = [n_examples]
            Predicted class labels after applying the Heaviside step function.
            Class labels are either 1 or -1.
        '''
        return np.where(self.net_input(X) >= 0.0, 1, -1)