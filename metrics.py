from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: 
    return (y_hat==y).mean()
    pass


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    TP = ((y == cls) & (y_hat == cls)).sum()
    FP = ((y != cls) & (y_hat == cls)).sum()
    if (TP+FP)>0:
        return TP/(TP+FP)
    else :
        return 0.0


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    TP = ((y_hat == cls) & (y == cls)).sum()
    FN = ((y_hat != cls) & (y == cls)).sum()
    if (TP+FN)>0:
        return TP/(TP+FN)
    else :
        return 0.0


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    rmse_ = ((y_hat - y) ** 2).sum() / y.size
    rmse_ = rmse_ ** 0.5
    return rmse_


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    mae_ = (y_hat - y).abs().sum() / y.size
    return mae_  
