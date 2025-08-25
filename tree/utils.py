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
    X = X.copy()
    for col in list(X.columns):  
        if check_ifreal(X[col]):
            continue

        k = X[col].nunique(dropna=True)

        if k >= 3:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=False, dummy_na=False)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

        elif k == 2:
            vals = sorted(X[col].dropna().unique(), key=lambda v: str(v))
            pos = vals[1]
            X[col] = (X[col] == pos).astype(int)



    return X


def check_ifreal(y: pd.Series, real_distinct_threshold: int = 16) -> bool:
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
    """
    Function to calculate the entropy

    entropy = -sum(p_i * log2(p_i))
    """
    if len(Y) == 0:
        return 0.0
    value_counts = Y.value_counts()
    total_count = Y.size
    prob = value_counts / total_count
    entropy_value = -np.sum(prob * np.log2(prob + 1e-10)) # Adding a small value to avoid log(0)
    return entropy_value


    


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index

    gini_index = 1 - sum(p_i^2)
    """
    value_counts = Y.value_counts()
    total_count = Y.size
    probs = value_counts / total_count
    gini_index_value = 1 - np.sum(probs ** 2)
    return gini_index_value


def mse(Y: pd.Series) -> float:
    """
    Function to calculate the mean squared error
    """
    if len(Y) == 0:
        return 0.0
    return np.mean((Y - Y.mean()) ** 2)


def check_criteria(Y: pd.Series, criterion: str):
    """
    Function to check if the criterion is valid
    """
    if criterion == "information_gain":
        if check_ifreal(Y):
            the_criterion = "mse"
        else:
            the_criterion = "entropy"
    elif criterion == "gini_index":
        the_criterion = "gini_index"
    else:
        raise ValueError(f"Unknown criterion: {criterion} it must be one of [information_gain, gini_index]")

    if the_criterion == "entropy":
        criterion_func = entropy
    elif the_criterion == "gini_index":
        criterion_func = gini_index
    elif the_criterion == "mse":
        criterion_func = mse
    else:
        raise ValueError(f"Unknown criterion: {the_criterion}")

    return the_criterion, criterion_func

def find_optimal_threshold(attr: pd.Series, Y: pd.Series, criterion: str) -> float | None:
    """
    Function to find the optimal threshold for a continuous attribute
    """

    my_criterion, criterion_func = check_criteria(Y, criterion)

    if attr.size <= 1:
        return None
    elif attr.size == 2:
        return attr.sum() / 2

    sorted_attr = attr.sort_values()


    split_points = (sorted_attr[:-1] + sorted_attr[1:]) / 2

    best_threshold = None
    best_information_gain = -np.inf

    for threshold in split_points:
        Y_left = Y[attr <= threshold]
        Y_right = Y[attr > threshold]

        if Y_left.empty or Y_right.empty:
            continue

        information_gain = criterion_func(Y) - ( (Y_left.size / Y.size) * criterion_func(Y_left) + (Y_right.size / Y.size) * criterion_func(Y_right))


        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_threshold = threshold

    return best_threshold

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    the_criterion, criterion_func = check_criteria(Y, criterion)

    # If the attribute is continuous, calculate the information gain using the MSE
    if check_ifreal(attr):
        threshold = find_optimal_threshold(attr, Y, criterion)
        if threshold is None:
            return 0.0
        Y_left = Y[attr <= threshold]
        Y_right = Y[attr > threshold]
        information_gain = criterion_func(Y) - ( (Y_left.size / Y.size) * criterion_func(Y_left) + (Y_right.size / Y.size) * criterion_func(Y_right))
        return information_gain

    # If the attribute is discrete, calculate the information gain for each unique value of the attribute
    total_criterion = criterion_func(Y)
    weighted_criterion = 0.0
    for value in attr.unique():
        Y_attr = Y[attr == value]
        weighted_criterion += (Y_attr.size / Y.size) * criterion_func(Y_attr)

    information_gain = total_criterion - weighted_criterion
    return information_gain

 





def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    best_attribute = None
    best_information_gain = -np.inf

    for feature in features:
        info_gain = information_gain(y, X[feature], criterion)
        if info_gain > best_information_gain:
            best_information_gain = info_gain
            best_attribute = feature

    return best_attribute


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
        X_left = X[X[attribute] <= value]
        X_right = X[X[attribute] > value]
    else:
        X_left = X[X[attribute] == value]
        X_right = X[X[attribute] != value]
    y_left = y.loc[X_left.index]
    y_right = y.loc[X_right.index]
    return X_left, y_left, X_right, y_right
