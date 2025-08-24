import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

X = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
y = pd.Series(y)

# 70% of data used for train and 30% of the data used for test
split = int(0.7 * X.shape[0]) #70% of the number of rows
X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nShape of X training data set is: ", X_train.shape)
print("\nShape of X testing data set is: ", X_test.shape)
print("\nShape of y training data set is: ", y_train.shape)
print("\nShape of y testing data set is: ", y_test.shape)

tree_model = DecisionTree(criterion='information_gain', max_depth=4)
tree_model.fit(X_train, y_train)

y_predicted = tree_model.predict(X_test)

print("\nCriteria used: Information Gain")
print("\nDepth taken: 4")

# Index are reseted for y_test and y_predicted and also for further functions so that error occuring due to index mismatching could be avoided
print("Accuracy of the model: ", np.round(accuracy(y_test.reset_index(drop=True), y_predicted.reset_index(drop=True)), 5))

# Precision and Recall for every class in decision tree

for cls in np.unique(y_test):
  print(f"Precision for class {cls} is {np.round(precision(y_test.reset_index(drop=True), y_predicted.reset_index(drop=True), cls), 5)}")
  print(f"Recall for class {cls} is {np.round(recall(y_test.reset_index(drop=True), y_predicted.reset_index(drop=True), cls), 5)}")