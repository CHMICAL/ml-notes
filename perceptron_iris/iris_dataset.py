import os
import pandas as pd

iris_dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(iris_dataset_url, header = None, encoding='utf-8')