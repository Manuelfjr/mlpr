"""
Utils module for classification models.
"""

from typing import Callable, Dict, Optional, Union

from numpy import array, linspace, ndarray
from pandas import DataFrame
from sklearn.base import BaseEstimator


def calculate_probas(
    fitted_models: Dict[str, BaseEstimator], X: Union[ndarray, DataFrame], proba_func: Optional[Callable] = None
) -> Dict[str, ndarray]:
    """
    Calculate probabilities for each fitted model.

    Parameters
    ----------
    fitted_models : dict
        A dictionary of fitted models. The keys are the model names and the values are the fitted models.
    X : array-like of shape (n_samples, n_features)
        The input samples.
    proba_func : callable, default=None
        A function to calculate the probabilities. If None, the default function is used.

    Returns
    -------
    probas : dict
        A dictionary of probabilities for each model. The keys are the model names and the values are the probabilities.
    """
    probas = {}
    for name, model in fitted_models.items():
        probs = model.predict_proba(X)
        if proba_func:
            probas[name] = proba_func(probs)
        else:
            probas[name] = 1 - probs.max(axis=1)
    return probas


def generate_colors(rgb_min: str, rgb_max: str, k: int) -> list:
    """
    Generate a list of colors.

    This function generates a list of colors by interpolating between two given RGB colors.

    Parameters
    ----------
    rgb_min : str
        The minimum RGB color (as a hexadecimal string).
    rgb_max : str
        The maximum RGB color (as a hexadecimal string).
    k : int
        The number of colors to generate.

    Returns
    -------
    list
        A list of interpolated colors in hexadecimal format.

    Examples
    --------
    >>> generate_colors("000000", "ffffff", 5)
    ['#000000', '#3f3f3f', '#7f7f7f', '#bfbfbf', '#ffffff']
    """
    rgb_min = array([int(rgb_min[i : i + 2], 16) for i in (0, 2, 4)])
    rgb_max = array([int(rgb_max[i : i + 2], 16) for i in (0, 2, 4)])
    diff = rgb_max - rgb_min
    colors = [rgb_min + diff * t for t in linspace(0, 1, k)]
    colors = [f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}" for color in colors]
    return colors
