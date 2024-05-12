"""
Module for surrogates model on machine learning.
"""

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from mlpr.ml.supervisioned.tunning.grid_search import GridSearch


class Surrogate(BaseEstimator, RegressorMixin, ClassifierMixin):
    """
    This class is used to train surrogate models to approximate black box models.

    Parameters
    ----------
    white_box : dict
        A dictionary of surrogate models.
    black_box : dict
        A dictionary of black box models.
    params_split : dict, optional
        Parameters for splitting the data. Defaults to None.
    normalize : bool, optional
        Whether to normalize the data. Defaults to True.
    params_norm : dict, optional
        Parameters for normalizing the data. Defaults to None.
    scoring : str, optional
        Scoring metric to use. Defaults to None.
    metrics : dict[str, Callable], optional
        Metrics for evaluating the models. Defaults to None.
    """

    def __init__(
        self,
        white_box: dict,
        black_box: dict,
        params_split: dict = None,
        normalize: bool = True,
        params_norm: dict = None,
        scoring: Optional[str] = None,
        metrics: Optional[dict[str, Callable]] = None,
    ):
        self.white_box = white_box
        self.black_box = black_box
        self.params_split = params_split
        self.params_norm = params_norm
        self.metrics = metrics
        self.normalize = normalize
        self.scoring = scoring
        self.best_model_black = None
        self.best_params_black = None
        self.best_model_white = None
        self.best_params_white = None
        self.grid_search_black = None
        self.grid_search_white = None

    def _grid_search(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
        models: dict[BaseEstimator, dict[str, any]],
    ) -> GridSearch:
        """
        Perform grid search to find the best model and parameters.

        Parameters
        ----------
        X : DataFrame or ndarray
            The training data.
        y : DataFrame, Series, or ndarray
            The target values.
        models : dict[BaseEstimator, dict[str, any]]
            The models and their parameters to search.

        Returns
        -------
        GridSearch
            The grid search object.
        """
        grid_search = GridSearch(
            X,
            y,
            params_split=self.params_split,
            models_params=models,
            normalize=self.normalize,
            params_norm=self.params_norm,
            scoring=self.scoring,
            metrics=self.metrics,
        )
        return grid_search

    def _fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ):
        """
        Fit the best black box and white box models.

        Parameters
        ----------
        X : DataFrame or ndarray
            The training data.
        y : DataFrame, Series, or ndarray
            The target values.
        """
        self.best_model_black.fit(X, y)
        self.best_model_white.fit(X, y)
        return self

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
        fitted_x: bool = False,
        **kwargs,
    ):
        """
        Fit the surrogate models to approximate the black box models.

        Parameters
        ----------
        X : DataFrame or ndarray
            The training data.
        y : DataFrame, Series, or ndarray
            The target values.
        fitted_x : bool, optional
            Whether X is already fitted with all data. Defaults to False.
        **kwargs
            Additional keyword arguments.
        """
        self.grid_search_black: GridSearch = self._grid_search(X, y, self.black_box)
        self.grid_search_black.search(**kwargs)
        self.best_model_black, self.best_params_black = self.grid_search_black.get_best_model()
        y_pred = self.best_model_black.predict(self.grid_search_black.X_test)

        self.grid_search_white: GridSearch = self._grid_search(self.grid_search_black.X_test, y_pred, self.white_box)
        self.grid_search_white.search(**kwargs)
        self.best_model_white, self.best_params_white = self.grid_search_white.get_best_model()
        if fitted_x:
            self._fit(X, y)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions from the black box and white box models.

        Parameters
        ----------
        X : DataFrame or ndarray
            The data to predict.

        Returns
        -------
        tuple
            The predictions from the black box and white box models.
        """
        black_box_predictions: np.ndarray = self.best_model_black.predict(X)
        white_box_predictions: np.ndarray = self.best_model_white.predict(X)

        return black_box_predictions, white_box_predictions
