import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ssl

np.random.seed(42)
# Gives exact same random results everytime

# Reading the data
ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

# we will predict for the column "mpg"

data = data.drop(columns='car name') 
# dropping car name column as it is just the label, not necessary for numeric calculation

data.replace("?", np.nan, inplace=True)
data = data.dropna(subset=["horsepower"])
# removing all rows where horsepower column is NaN

data["horsepower"] = data["horsepower"].astype(float)
data = data.drop_duplicates() # removing duplicates

X = data.drop(columns='mpg')
y = data['mpg']

print("\nX-shape is: ", X.shape)
print("\ny-shape is: ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nX Train data size: ", X_train.shape)
print("\nX Test data size: ", X_test.shape)
print("\ny Train data size: ",y_train.shape)
print("\ny Test data size: ", y_test.shape)

# Decision Tree Implementation

my_dtree = DecisionTree(criterion='information_gain', max_depth=5)
my_dtree.fit(X_train, y_train)

y_prediction = my_dtree.predict(X_test)
print("\nCustom Decision Tree performance is: \n")
print(f"Root mean squared error for the above data is: {np.round(rmse(y_prediction, y_test), 5)}")
print(f"Mean absolute error for the above data is: {np.round(mae(y_prediction, y_test), 5)}")

# Sklearn decision tree

dt_sklearn = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_sklearn.fit(X_train, y_train)
y_prediction_sklearn = dt_sklearn.predict(X_test)

print("\nSklearn decision tree performace is: \n")
print(f"Root mean squared error for the above data is: {np.round(np.sqrt(mean_squared_error(y_test, y_prediction_sklearn)), 5)}")
print(f"Mean absolute error for the above data is: {np.round(mean_absolute_error(y_test, y_prediction_sklearn), 5)}")