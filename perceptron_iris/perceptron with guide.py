import numpy as np


'''
Perceptron classifier
-----------------------
The perceptron classifier is a type of supervised learning algorithm used for binary classification.
The perceptron classifier makes predictions by assigning weights to input features and taking a weighted sum (multiplying 
each input feature by a corresponding weight value and adding up the results). 
It then applies a threshold function to the sum to produce a prediction of either 0 or 1.
It is an effective way to find a decision boundary between two classes of data(-1, 1).

1. Initialize the weights: 
    Weights are initialized as small random values. Typically drawn from a normal distribution with a mean 0, 
    and a small standard deviation (e.g., 0.01). Weights are essentially the parameters that establish the decision boundaries between the two classes
    of data, and correspond to features of the dataset. The importance of each weight determines the prediction of the perceptron. When the training
    is ongoing, these weights will be continually updated to find the best decision boundary.

2. Calculate the net input:
    For each training example (e.g., In house price prediction model, for each house...) calculate the net input.
    If we have a training set with n examples and m features, the weight vector w is a 1D array of length m+1 
    (where the extra dimension corresponds to the bias term). Each element of the weight vector w is multiplied
    by its corresponding feature in the input data, and the results are summed up to produce the net input to the
    perceptron.
        1. Multiply the features by their corresponding weights
        2. Sum the results

3. Apply the activation function (which is the threshold function in perceptron):
    The net input is then passed through the threshold function, which returns either 1 or -1 depending on whether the net input is above
    or below a certain threshold.

4. Update the weights:
    If the predicted output does not match the desired output for a given training example, the weights are updated to correct the prediction. 
    This is done by adding or subtracting the product of the learning rate (learning_rate) and the error to each weight. The learning rate will decide how much
    the weights will be updated in every iteration. This is usually set to a small value. If it is too large, the algorithm can fail to converge. If it is too
    small, it may take a long time for the algorithm to converge. 

Repeat these 4 steps for every training example until the algorithm converges (i.e., Is able to accurately predict ALL the training examples
or has at least achieved a very low error rate) or the maximum number of iterations(n_iter) is reached. The maximum number of iterations is ultimately decided by the
user, but should be high enough to give the algorithm enough time to converge. However, it should not be excessively high, as this will cause the
algorithm to potentially run for a very long time.

5. Return weights:
    Once the algorithm has converged or the maximum number of iterations (n_iter) is reached, the weights are returned. These weights can then be used to predict
    classifications for unseen data. 

'''

class Perceptron_Classifier(object):
    '''
    Parameters
    -------------
    learning_rate : float
        Learning rate (between 0.0 and 1.0). Decides how much the weights are updated in each iter. (typically, small).
    n_iter : int
        Number of iterations over the training dataset. 
    random_seed : int
        Random number generator seed for random weight initialization (Usually a small standard deviation from a normal distribution of mean 0 (e.g., 0.01))
        By setting a seed value for the random number generator used to initialize the weights, the same set of random numbers will be used each time 
        the model is trained. 
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
        Fit training data. This is essentially the training process of the Perceptron model
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