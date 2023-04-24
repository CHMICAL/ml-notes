import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from iris_dataset import df as iris_df
from perceptron_classifier import Perceptron_Classifier
from matplotlib.colors import ListedColormap

'''
Training the Perceptron classifier...
'''

# Select setosa and versicolor and transform them to 1s (versicolor) and -1s (setosa)
y = iris_df.iloc[0:100, 4].values
y = np.where(y=='Iris-setosa', -1, 1)

# Extract features (sepal-0 and petal length-2)
X = iris_df.iloc[0:100, [0,2]].values

ppn = Perceptron_Classifier(learning_rate=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates to weights (misclassifications)')

plt.savefig('training data.png', dpi=300)
# plt.show()

# From this plot, we can see that after the 6th epoch, the model converges (i.e., weights no longer need to be updated)

plt.clf()

# To more clearly visualise the convergence, let's plot the decision boundaries. The following function will plot decision boundaries for 2D datasets.

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')


plt.savefig('decision_boundaries.png', dpi=300)
# plt.show()


# It is worth noting that the Perceptron classifier will converge if there is a clear decision boundary between the two classifications.
# Without this, convergence becomes a big problem with perceptron. It is therefore efficient and effective for simple datasets, but not
# effective for more complex datasets. 