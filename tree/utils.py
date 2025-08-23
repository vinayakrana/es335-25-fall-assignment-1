"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    for col in X.columns:
        if X[col].dtype == "object":
            X = pd.concat([X, pd.get_dummies(X[col], prefix=col,dtype=int)], axis=1)
            X.drop(col, axis=1, inplace=True)
    return X

    pass

def check_ifreal(y: pd.Series, threshold = 0.1) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    unique_nums = y.nunique()
    total_nums = len(y)
    if(unique_nums/total_nums < threshold):
        return False
    return True


def entropy(Y: pd.Series) -> float:
    uniques = Y.value_counts()
    total = np.size(Y)
    p = uniques/total
    return np.sum(-p*np.log2(p))


def gini_index(Y: pd.Series) -> float:
    uniques = Y.value_counts()
    total = np.size(Y)
    p = uniques/total
    return (1 - np.sum(p**2))

def MSE(Y: pd.Series)->float:
    """
    Function to calculate the MSE
    """
    y_mean=np.mean(Y)
    ans=np.sum((Y-y_mean)**2)/len(Y)
    return ans

 
def check_criteria(Y: pd.Series, criterion: str) -> None:
    """
    Function to check which criterion to use out of the 4 possible conditions of I/P and O/P
    """
    fn = None
    # (, Discrete)
    if (check_ifreal(Y)==False):
        # Using Entropy/GiniIndex
        if (criterion=='entropy'):
            fn = entropy(Y)
            return "entropy",fn
        elif (criterion=='gini index'):
            fn = gini_index(Y)
            return "gini",fn

    # (, Real)
    else:
        fn = MSE(Y)
        return "MSE",fn
       
    

def find_optimal_threshold(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to find the optimal threshold for a real feature

    Returns the threshold value for best split in a given real feature
    """
    sorted_attr = attr.sort_values()
    if sorted_attr.size == 1:
        return None
    elif sorted_attr.size == 2:
        return (sorted_attr.sum()) / 2
    split_points = (sorted_attr[:-1] + sorted_attr[1:]) / 2
    
    best_threshold = None
    best_gain = -np.inf

    for threshold in split_points:
        Y_left = Y[attr <= threshold]
        Y_right = Y[attr > threshold]

        if Y_left.empty or Y_right.empty:
            continue

        total_criterion = Y_left.size / Y.size * check_criteria(Y_left, criterion)[1] + Y_right.size / Y.size * check_criteria(Y_right,criterion)[1]
        information_gain_value = check_criteria(Y,criterion)[1] - total_criterion

        if information_gain_value > best_gain:
            best_threshold = threshold
            best_gain = information_gain_value

    return best_threshold



def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion_fn (entropy, gini index or MSE)
    """
    criterion_fn = check_criteria(Y, criterion)[1]

    # Attribute is Real
    if (check_ifreal(attr)):
        threshold = find_optimal_threshold(Y, attr, criterion)
        if threshold is None:
            return 0
        top = Y[attr <= threshold]
        top_p = top.size()/Y.size()
        bottom = Y[attr > threshold]
        bottom_p = bottom.size()/Y.size()
        return criterion_fn(Y) - (top_p*criterion_fn(top) + bottom_p*criterion_fn(bottom))

    # Attribute is Discrete
    else:
        weighted_H = 0
        for unique in attr.unique():
            sub_attr = Y[attr==unique]
            sub_attr_p = np.size(sub_attr)/np.size(attr)
            weighted_H += sub_attr_p*criterion_fn(sub_attr)
        return criterion_fn(Y) - weighted_H



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    best_info_gain = 0
    opt_split = None

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    for feature in features:
        current_gain = information_gain(y, X[feature], criterion)
        if(current_gain > best_info_gain):
            best_info_gain = current_gain
            opt_split = feature

    return opt_split, best_info_gain

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    if check_ifreal(X[attribute]):
        left_X = X[X[attribute] <= value]
        right_X = X[X[attribute] > value]
    else:
        left_X = X[X[attribute] == value]
        right_X = X[X[attribute] != value]

    left_y = y[left_X.index]
    right_y = y[right_X.index]

    return left_X, right_X, left_y, right_y

    
