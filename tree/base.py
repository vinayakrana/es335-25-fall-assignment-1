"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal,Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class TreeNode:
    attribute: str #this will store input feature on which split will occur
    value: Union[str, float] #this will store the value of split in the input feature either as discrete like (Sunny) or real value like (40)
    left: "TreeNode"
    right: "TreeNode"
    is_leaf: bool
    output: Union[str, float]
    Impurity_measure=str
    criterion_val=float
    gain: float
    
    def __init__(self, attribute=None, value=None, left=None, right=None, is_leaf=False, output=None, Impurity_measure=None,criterion_val=None, gain=0):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.output = output
        self.Impurity_measure=Impurity_measure
        self.criterion_val = criterion_val
        self.gain = gain
    # def is_leaf_node(self):
    #     return self.is_leaf


class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.Tree=None

    def fit(self, X: pd.DataFrame, y: pd.Series,depth: int = 0) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 
        def Tree_construct(X: pd.DataFrame, y: pd.Series, depth: int)->TreeNode:
            Impurity_measure,criterion_val=check_criteria(y)
           
           # Create a leaf node if max_depth is reached or if all target values are identical.
            if depth >= self.max_depth or y.nunique() == 1:
                if check_ifreal(y):
                    return TreeNode(is_leaf=True, output=np.round(y.mean(),4), Impurity_measure=Impurity_measure,criterion_val=criterion_val)
                else:
                    return TreeNode(is_leaf=True, output=y.mode()[0], Impurity_measure=Impurity_measure,criterion_val=criterion_val)
            
            best_attribute = opt_split_attribute(X, y,self.criterion,X.columns)[0]
          # Create a leaf node if no valid attribute is available for splitting.
            if best_attribute is None:
                if check_ifreal(y):
                    return TreeNode(is_leaf=True, output=np.round(y.mean(),4),Impurity_measure=Impurity_measure,criterion_val=criterion_val)
                else:
                    return TreeNode(is_leaf=True, output=y.mode()[0], Impurity_measure=Impurity_measure,criterion_val=criterion_val)

            if check_ifreal(X[best_attribute]):
                best_value = find_optimal_threshold(y, X[best_attribute], self.criterion)
                # create a leaf node, if there is no valid threshold.
                if best_value is None:
                    if check_ifreal(y):
                        return TreeNode(is_leaf=True, output=np.round(y.mean(),4),Impurity_measure=Impurity_measure,criterion_val=criterion_val)
                    else:
                        return TreeNode(is_leaf=True, output=y.mode()[0],Impurity_measure=Impurity_measure,criterion_val=criterion_val)
            else:
                best_value = X[best_attribute].mode()[0]

            left_X, right_X, left_y, right_y = split_data(X, y, best_attribute, best_value)

            # Create a leaf node if a valid split cannot be made.
            if left_X.empty or right_X.empty:
                if check_ifreal(y):
                    return TreeNode(is_leaf=True, output=np.round(y.mean(),4), Impurity_measure=Impurity_measure,criterion_val=criterion_val)
                else:
                    return TreeNode(is_leaf=True, output=y.mode()[0], Impurity_measure=Impurity_measure,criterion_val=criterion_val)
                
            left_subtree = Tree_construct(left_X, left_y, depth + 1)
            right_subtree = Tree_construct(right_X, right_y, depth + 1)

            return TreeNode(attribute=best_attribute, value=best_value, left=left_subtree, right=right_subtree, Impurity_measure=Impurity_measure,criterion_val=criterion_val, gain=information_gain(y, X[best_attribute], self.criterion))
        self.Tree = Tree_construct(X, y, depth)

 

        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        
        def predict_row(x: pd.Series) -> float:
            curr_node = self.tree
            while (not curr_node.is_leaf_node()):
                if (check_ifreal(x[curr_node.attribute])):
                    if (x[curr_node.attribute] <= curr_node.value):
                        curr_node = curr_node.left
                    else:
                        curr_node = curr_node.right
                else:
                    if (x[curr_node.attribute] == curr_node.value):
                        curr_node = curr_node.left
                    else:
                        curr_node = curr_node.right
            return curr_node.output
        return pd.Series([predict_row(x) for _,x in X.iterrows()])

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass
