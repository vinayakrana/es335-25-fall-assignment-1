from typing import Union
import pandas as pd



def validate_inputs(y_hat, y):
    # Ensure both y_hat and y are pandas Series
    assert isinstance(y_hat, pd.Series), "y_hat must be a pandas Series"
    assert isinstance(y, pd.Series), "y must be a pandas Series"
    assert y_hat.size == y.size, "y_hat and y must be the same size"

    y_hat = y_hat.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return y_hat, y


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    y_hat, y = validate_inputs(y_hat, y)
    accuracy_value = (y_hat == y).sum() / y.size
    return accuracy_value


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    y_hat, y = validate_inputs(y_hat, y)
    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    y_hat, y = validate_inputs(y_hat, y)
    tp = ((y_hat == cls) & (y == cls)).sum()
    fn = ((y_hat != cls) & (y == cls)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    y_hat, y = validate_inputs(y_hat, y)
    rmse_value = ((y_hat - y) ** 2).sum() / y.size
    rmse_value = rmse_value ** 0.5
    return rmse_value


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    y_hat, y = validate_inputs(y_hat, y)
    mae_value = (y_hat - y).abs().sum() / y.size
    return mae_value 
