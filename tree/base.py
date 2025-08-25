"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

from dataclasses import dataclass
from typing import Literal, Union
from graphviz import Digraph
from IPython.display import Image, display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Union
from tree.utils import *

np.random.seed(42)


@dataclass
class Node:
    attribute: str
    value: float
    left: "Node"
    right: "Node"
    is_leaf: bool
    output: Union[str, float]
    criterion_pair: tuple
    gain: float

    def __init__(self, attribute: str, value: float, left: "Node", right: "Node", is_leaf: bool, output: None, criterion_pair: tuple, gain: float):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.output = output
        self.criterion_pair = criterion_pair
        self.gain = gain
    
    def is_leaf_node(self):
        return self.is_leaf


class DecisionTree:
    
    
    criterion: Literal["information_gain", "gini_index"]  
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None  # Root node of the tree

    
    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth=0) -> Node:
        """
        Function to implement the decision tree algorithm recursively
        """
        my_criterion, criterion_func = check_criteria(y, self.criterion)
        criterion_value = criterion_func(y)
       

        if depth == self.max_depth or y.nunique() == 1:
            if check_ifreal(y):
                return Node(
                    attribute=None,
                    value=None,
                    left=None,
                    right=None,
                    is_leaf=True,
                    output=y.mean(),
                    criterion_pair=(my_criterion, criterion_value),
                    gain=0.0
                )
            else:
                return Node(
                    attribute=None,
                    value=None,
                    left=None,
                    right=None,
                    is_leaf=True,
                    output=y.mode()[0],
                    criterion_pair=(my_criterion, criterion_value),
                    gain=0.0
                )

        best_split_attribute = opt_split_attribute(X, y, self.criterion, X.columns)
        if best_split_attribute is None:
            if check_ifreal(y):
                return Node(
                    attribute=None,
                    value=None,
                    left=None,
                    right=None,
                    is_leaf=True,
                    output=y.mean(),
                    criterion_pair=(my_criterion, criterion_value),
                    gain=0.0
                )
            else:
                return Node(
                    attribute=None,
                    value=None,
                    left=None,
                    right=None,
                    is_leaf=True,
                    output=y.mode()[0],
                    criterion_pair=(my_criterion, criterion_value),
                    gain=0.0
                )

        else:
            if check_ifreal(X[best_split_attribute]):
                best_split_value = find_optimal_threshold(X[best_split_attribute], y, self.criterion)
                if best_split_value is None:
                    if check_ifreal(y):
                        return Node(
                            attribute=None,
                            value=None,
                            left=None,
                            right=None,
                            is_leaf=True,
                            output=y.mean(),
                            criterion_pair=(my_criterion, criterion_value),
                            gain=0.0
                        )
                    else:
                        return Node(
                            attribute=None,
                            value=None,
                            left=None,
                            right=None,
                            is_leaf=True,
                            output=y.mode()[0],
                            criterion_pair=(my_criterion, criterion_value),
                            gain=0.0
                        )
            else:
                best_split_value = X[best_split_attribute].mode()[0]
            
            X_left, y_left, X_right, y_right = split_data(X, y, best_split_attribute, best_split_value)

            if X_left.empty or X_right.empty:
                if check_ifreal(y):
                    return Node(
                        attribute=None,
                        value=None,
                        left=None,
                        right=None,
                        is_leaf=True,
                        output=y.mean(),
                        criterion_pair=(my_criterion, criterion_value),
                        gain=0.0
                    )
                else:
                    return Node(
                        attribute=None,
                        value=None,
                        left=None,
                        right=None,
                        is_leaf=True,
                        output=y.mode()[0],
                        criterion_pair=(my_criterion, criterion_value),
                        gain=0.0
                    )

            left_child = self._build_tree(X_left, y_left, depth + 1)
            right_child = self._build_tree(X_right, y_right, depth + 1)

            return Node(
                attribute=best_split_attribute,
                value=best_split_value,
                left=left_child,
                right=right_child,
                is_leaf=False,
                output=None,
                criterion_pair=(my_criterion, criterion_value),
                gain=information_gain(y, X[best_split_attribute], self.criterion)
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 


        self.tree = self._build_tree(X, y)


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        
        predictions = []
        for _, row in X.iterrows():
            curr_node = self.tree
            while not curr_node.is_leaf:
                if check_ifreal(X[curr_node.attribute]):
                    if row[curr_node.attribute] <= curr_node.value:
                        curr_node = curr_node.left
                    else:
                        curr_node = curr_node.right
                else:
                    if row[curr_node.attribute] == curr_node.value:
                        curr_node = curr_node.left
                    else:
                        curr_node = curr_node.right
            predictions.append(curr_node.output)
        return pd.Series(predictions)

    def plot(self, path = None) -> None:
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
        dot = Digraph()
        if not self.tree:
            print("Tree not trained yet")
            return
        
        def print_tree(self) -> str:
            def recurse(node: Node, depth: int) -> str:
                output = ''
                if node.is_leaf:
                    output += f'Class: {node.output}\n'
                else:
                    output += f'?(attribute :  {node.attribute} <= {node.value:.2f})\n'
                    output += f"{'  ' * (depth + 1)}Y: "
                    output += recurse(node.left, depth + 1)
                    output += f"{'  ' * (depth + 1)}N: "
                    output += recurse(node.right, depth + 1)
                return output

            

            if not self.tree:
                return "Tree not trained yet"
            else:
                return recurse(self.tree, 0)

        def __repr__(self):
            return f"DecisionTree(criterion={self.criterion}, max_depth={self.max_depth})\n\nTree Structure:\n{self.print_tree()}"
        
        def add_node(node: Node, parent_name: str = None, edge_label: str = None) -> None:
            node_id = str(id(node))
            if node.is_leaf:
                node_label = f"Prediction: {node.output}\n {node.criterion_pair[0]} = {node.criterion_pair[1]:.4f}"
            else:
                node_label = f"(attribute : {node.attribute} <= {node.value:.2f})\n {node.criterion_pair[0]} = {node.criterion_pair[1]:.4f}"
            dot.node(node_id, label=node_label, shape='box' if node.is_leaf else 'ellipse')

            if parent_name:
                dot.edge(parent_name, node_id, label=edge_label)
            if node.left:
                add_node(node.left, node_id, edge_label="Yes")
            if node.right:
                add_node(node.right, node_id, edge_label="No")

        add_node(self.tree)
        print("\nDecision Tree Structure:")
        print(print_tree(self))
        if path:
            dot.render(path, format="png", view=False, cleanup=True)
            display(Image(filename=f"{path}.png"))  
        else:
            png_data = dot.pipe(format='png')
            display(Image(data=png_data))

        

        