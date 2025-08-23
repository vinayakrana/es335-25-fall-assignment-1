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




def check_ifreal(y: pd.Series, real_distinct_threshold =15) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if pd.api.types.is_categorical_dtype(y):
        return False
    if pd.api.types.is_bool_dtype(y):
        return False
    if pd.api.types.is_float_dtype(y):
        return True
    if pd.api.types.is_integer_dtype(y):
        return len(y.unique()) > real_distinct_threshold
    if pd.api.types.is_string_dtype(y):
        return False
    return False

def entropy(Y: pd.Series) -> float:
    value_counts = Y.value_counts()/Y.size
    probs = value_counts[value_counts > 0]  # ignore zeros
    return -np.sum(probs * np.log2(probs))


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
    ans=np.sum((Y-y_mean)**2)/Y.size
    return ans

 
def check_criteria(Y: pd.Series, criterion: str):
    """
    Function to check which criterion to use out of the 4 possible conditions of I/P and O/P
    """
    fn = None
    # (, Discrete)
    if not check_ifreal(Y):  # Discrete output case
        if criterion == 'information_gain' or criterion == 'entropy':
            return "entropy", entropy
        elif criterion == "gini_index": 
            return "gini_index", gini_index
        else:
            raise ValueError(f"Unknown criterion for classification: {criterion}")
    # (, Real)
    else:
        return "MSE", MSE


def find_optimal_threshold(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to find the optimal threshold for a real feature

    Returns the threshold value for best split in a given real feature
    """
    my_criterion, criterion_func = check_criteria(Y, criterion)

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

        total_criterion = Y_left.size / Y.size * criterion_func(Y_left) + Y_right.size / Y.size * criterion_func(Y_right)
        information_gain_value = criterion_func(Y) - total_criterion

        if information_gain_value > best_gain:
            best_threshold = threshold
            best_gain = information_gain_value

    return best_threshold



def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion_fn (entropy, gini index or MSE)
    """
    my_criterion, criterion_func = check_criteria(Y, criterion)

    # Attribute is Real
    if (check_ifreal(attr)):
        threshold = find_optimal_threshold(Y, attr, criterion)
        if threshold is None:
            return 0
        top = Y[attr <= threshold]
        top_p = top.size/Y.size
        bottom = Y[attr > threshold]
        bottom_p = bottom.size/Y.size
        return criterion_func(Y) - (top_p*criterion_func(Y) + bottom_p*criterion_func(Y))

    # Attribute is Discrete
    else:
        weighted_H = 0
        for unique in attr.unique():
            sub_attr = Y[attr==unique]
            sub_attr_p = np.size(sub_attr)/np.size(attr)
            weighted_H += sub_attr_p*criterion_func(Y)
        return criterion_func(Y) - weighted_H



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    best_feature = None
    best_gain = -np.inf

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    for feature in features:
        current_gain = information_gain(y, X[feature], criterion)
        if(current_gain > best_gain):
            best_feature = feature
            best_gain = current_gain

    return best_feature

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

    left_y = y.loc[left_X.index]
    right_y = y.loc[right_X.index]

    return left_X, left_y, right_X, right_y

    
