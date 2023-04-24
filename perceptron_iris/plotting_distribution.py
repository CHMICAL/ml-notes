import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from iris_dataset import df as iris_df
from perceptron_classifier import Perceptron_Classifier

'''
Plotting distribution of flower examples (iris sertosa and iris versicolor) against their petal and sepal lengths.
'''


# Select setosa and versicolor and transform them to 1s (versicolor) and -1s (setosa)
y = iris_df.iloc[0:100, 4].values
y = np.where(y=='Iris-setosa', -1, 1)

# Extract features (sepal-0 and petal length-2)
X = iris_df.iloc[0:100, [0,2]].values

# Plotting setosa and versicolor petal (y axis) and sepal (x axis) length
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.legend(loc='upper left')

plt.savefig('distribution of sertosa and versicolor.png', dpi=300)
# plt.show()

# From the plot, we can see the distribution of flower examples. We can also see that a linear classification should be sufficient to
# classify versicolors and sertosas...