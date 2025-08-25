import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tree.base import DecisionTree
from metrics import *
import time
from sklearn.model_selection import KFold
from itertools import product
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
def create_data():
    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
    return X, y

def plot_data(X, y):
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)


    # For plotting
    plt.title("Scatter plot of the synthetic dataset")
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Scatter plot of the synthetic dataset")
    plt.show()
   

# Write the code for Q2 a) and b) below. Show your results.

# Q2 a)
print("Q2 a)")

def create_data_train_model(X,y):
    X = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    y = pd.Series(y, name='Target')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)

    model = DecisionTree(criterion='information_gain', max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Criteria :", "Information Gain")
    print("Accuracy :", np.round(accuracy(y_test, y_pred),4))




# Q2 b)

print("\nQ2 b)")

# K fold cross Validation as a function
def k_fold_cross_validation(X, y, model_class, folds=5, random_state=42, criterion='information_gain', max_depth=5):
    fold_size = X.shape[0] // folds
    accuracies = []
    for fold in range(folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold != folds - 1 else X.shape[0]
        X_test = X.iloc[start:end]
        y_test = y.iloc[start:end]
        X_train = pd.concat([X.iloc[:start], X.iloc[end:]])
        y_train = pd.concat([y.iloc[:start], y.iloc[end:]])
        model = model_class(criterion=criterion, max_depth=max_depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = np.round(accuracy(y_test, y_pred), 4)
        print(f"Fold {fold+1}/{folds}")
        print("Criteria :", criterion)
        print("Accuracy :", acc)
        accuracies.append(acc)
    print("\nAverage Accuracy:", np.round(np.mean(accuracies), 4))
    






def nested_cross_validation(X, y):
    # Define hyperparameters
    hyperparameters = {
        'criterion': ['information_gain', 'gini_index'], 
        'max_depth': [1,2,3,4,5,6,7,8,9,10]
    }

    outer_folds, inner_folds = 5, 5
    kf_outer = KFold(n_splits=outer_folds, shuffle=False)
    kf_inner = KFold(n_splits=inner_folds, shuffle=False)

    results = []
    outer_count = 0

    for outer_train_idx, outer_test_idx in kf_outer.split(X):
        X_outer_train, X_outer_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_outer_train, y_outer_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

        inner_count = 0

        for inner_train_idx, inner_val_idx in kf_inner.split(X_outer_train):
            print(f"Outer Fold: {outer_count+1}, Inner Fold: {inner_count+1}")

            X_inner_train, X_inner_val = X_outer_train.iloc[inner_train_idx], X_outer_train.iloc[inner_val_idx]
            y_inner_train, y_inner_val = y_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_val_idx]

            for max_depth in hyperparameters['max_depth']:
                for criterion in hyperparameters['criterion']:
                    model = DecisionTree(max_depth=max_depth, criterion=criterion)
                    model.fit(X_inner_train, y_inner_train)
                    y_pred = model.predict(X_inner_val)
                    acc = accuracy(y_inner_val, y_pred)

                    results.append({
                        'outer_fold': outer_count,
                        'inner_fold': inner_count,
                        'max_depth': max_depth,
                        'criterion': criterion,
                        'validation_accuracy': acc
                    })

            inner_count += 1
        outer_count += 1

    return pd.DataFrame(results), outer_folds, hyperparameters


def analyze_results(final_results, outer_folds, hyperparameters):
    fig, ax = plt.subplots(1, outer_folds, figsize=(20, 5))
    for i in range(outer_folds):
        outer_fold_results = final_results[final_results['outer_fold'] == i]

        for criterion in hyperparameters['criterion']:
            criterion_results = outer_fold_results[outer_fold_results['criterion'] == criterion]
            accuracies = criterion_results.groupby('max_depth')['validation_accuracy'].mean()
    
            ax[i].plot(accuracies, label=criterion)
            ax[i].set_title(f'Outer Fold {i}')
            ax[i].set_xlabel('Max Depth')
            ax[i].set_ylabel('Validation Accuracy')
            ax[i].legend()

    best_depths = {criterion: [] for criterion in hyperparameters['criterion']}
    

    for i in range(outer_folds):

        for criterion in hyperparameters['criterion']:
            outer_fold_df = final_results[final_results["outer_fold"] == i]
          
            top_results = outer_fold_df.groupby(['max_depth', 'criterion']).mean()["validation_accuracy"].sort_values(ascending=False)
         
            best_depths[criterion].append(int(top_results.idxmax()[0]))

    print("Best Depths: ", best_depths)
    print("Mean Best Depth: ")
    for criterion in hyperparameters['criterion']:
        print(f"Criterion: {criterion}, Mean Best Depth: {np.mean(best_depths[criterion])}")