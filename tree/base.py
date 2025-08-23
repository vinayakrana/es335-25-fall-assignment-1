"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *
from graphviz import Digraph
from IPython.display import Image, display

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        pass

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

        if not self.tree:
            print("Tree is not created yet")
            return 
        
        root = Digraph()

        def addnode(node: TreeNode, parentName: str = None, edge_label: str = None):
            node_id = str(id(node))
            # id returns unique id which helps graphviz to identify the node

            if node.is_leaf:
                node_label = f"Prediction: {node.output}"
            else:
                node_label = f"?(attr {node.attribute} <= {node.value})"
            root.node(node_id, label=node_label, shape='box' if node.is_leaf else 'ellipse')
            if parentName:
                root.edge(parentName, node_id, label = edge_label)
            if node.left:
                addnode(node.left, node_id, 'Yes')
            if node.right:
                addnode(node.right, node_id, 'No')
        
        addnode(self.tree)
        print("\nDecision Tree Data-Structure: ")
        print(self.print_tree())

        if path:
            root.render(path, format = 'png', view = False, cleanup = False)
            display(Image(filename=f"{path}.png"))
        else:
            png_data = root.pipe(format='png')
            display(Image(data = png_data))
    
    def printTree(self)->str:
        def print_node(node: TreeNode, indent: str = ""):
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
        
        if not self.tree:
            return "Tree is not created yet"
        else:
            return print_node(self.tree)