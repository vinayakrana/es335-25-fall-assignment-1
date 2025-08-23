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
import graphviz
from graphviz import Digraph
from IPython.display import Image, display

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
            Impurity_measure, criterion_function=check_criteria(y, self.criterion)
            criterion_val = criterion_function(y)

           # Create a leaf node if max_depth is reached or if all target values are identical.
            if depth >= self.max_depth or y.nunique() == 1:
                if check_ifreal(y):
                    return TreeNode(is_leaf=True, output=np.round(y.mean(),4), Impurity_measure=Impurity_measure,criterion_val=criterion_val)
                else:
                    return TreeNode(is_leaf=True, output=y.mode()[0], Impurity_measure=Impurity_measure,criterion_val=criterion_val)
            
            best_attribute = opt_split_attribute(X, y, self.criterion, X.columns)
          # Create a leaf node if no valid attribute is available for splitting.
            if best_attribute is None:
                if check_ifreal(y):
                    return TreeNode(is_leaf=True, output=np.round(y.mean(),4), Impurity_measure=Impurity_measure, criterion_val=criterion_val)
                else:
                    return TreeNode(is_leaf=True, output=y.mode()[0], Impurity_measure=Impurity_measure, criterion_val=criterion_val)

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

            left_X, left_y, right_X, right_y = split_data(X, y, best_attribute, best_value)

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
        
        def predict_row(x: pd.Series) -> float:
            curr_node = self.Tree
            while not curr_node.is_leaf:
                if check_ifreal(x[curr_node.attribute]):
                    if x[curr_node.attribute] <= curr_node.value:
                        curr_node = curr_node.left
                    else:
                        curr_node = curr_node.right
                else:
                    if x[curr_node.attribute] == curr_node.value:
                        curr_node = curr_node.left
                    else:
                        curr_node = curr_node.right
            return curr_node.output

        # Important: return the predictions as a Series
        return pd.Series([predict_row(x) for _, x in X.iterrows()])


    def plot(self, path = None) -> None: 
        # if path is given, saves tree png at that location
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

        if not self.Tree:
            print("Tree is not created yet")
            return 
        
        root = Digraph()

        def addnode(node: TreeNode, parentName: str = None, edge_label: str = None) -> None:
            node_id = str(id(node))
            # id returns unique id which helps graphviz to identify the node

            if node.is_leaf:
                node_label = f"Prediction: {node.output}\n {node.Impurity_measure} = {node.criterion_val:.4f}"
            else:
                node_label = f"?(attr {node.attribute} <= {node.value:.2f})\n {node.Impurity_measure} = {node.criterion_val:.4f}"
            root.node(node_id, label=node_label, shape='box' if node.is_leaf else 'ellipse')
            if parentName:
                root.edge(parentName, node_id, label = edge_label)
            if node.left:
                addnode(node.left, node_id, 'Yes')
            if node.right:
                addnode(node.right, node_id, 'No')
        
        addnode(self.Tree)
        print("\nDecision Tree Data-Structure: ")
        print(self.printTree())

        if path:
            root.render(path, format = 'png', view = False, cleanup = False)
            display(Image(filename=f"{path}.png"))
        else:
            png_data = root.pipe(format='png')
            display(Image(data = png_data))
    
    def printTree(self) -> str:
        def print_node(node: TreeNode, indent: str = "") -> str:
            output = ""
            if node.is_leaf:
                output+=f"Class: {node.output}\n"
            else:
                output+=f"?attr{node.attribute}<={node.value:.4f}\n"
                output+= indent+"   Yes: "
                output+= print_node(node.left, indent+"     ")
                output+= indent+"   No: "
                output+= print_node(node.right, indent+"    ")
            return output
        
        if not self.Tree:
            return "Tree is not created yet"
        else:
            return print_node(self.Tree)
        
# def _repr__(self):
#     return f"DecisionTree(criterion=)"