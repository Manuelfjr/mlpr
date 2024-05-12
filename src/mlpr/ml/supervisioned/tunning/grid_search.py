"""
Module for performing grid search on machine learning models.
"""

import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class GridSearch:  # pylint: disable=too-many-instance-attributes
    """
    Class for performing grid search on machine learning models.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models_params: Dict[BaseEstimator, Dict[str, Any]],
        params_split: dict = None,
        normalize: bool = True,
        params_norm: dict = None,
        scoring: Optional[str] = None,
        metrics: Optional[Dict[str, Callable]] = None,
    ) -> None:
        """
        Initialize the GridSearch object.

        Parameters
        ----------
        X : np.ndarray
            Features matrix.
        y : np.ndarray
            Target vector.
        models_params : dict
            Dictionary with models and parameters to search.
        params_split : dict, default={}
            Parameters for train-test split. Could include 'test_size', 'random_state', etc.
        normalize : bool, default=True
            Whether to normalize the data.
        params_norm : dict, default={}
            Parameters for the normalization process.
        scoring : str, default=None
            Scoring metric to evaluate the models. Must be a valid scoring metric for sklearn's GridSearchCV.
        """
        if params_split is None:
            params_split = {}
        if params_norm is None:
            params_norm = {}
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(X, y, **params_split)
        self.models_params: Dict[BaseEstimator, Dict[str, Any]] = models_params
        self.fitted = {}
        self.metrics = metrics if metrics else {}

        if normalize:
            self.normalize_data(**params_norm)

        self.best_model = None
        self.best_params = None
        self.scoring: str = scoring
        self._scores = None
        self._metrics = None
        self._params = None

    def split_data(
        self, X: np.ndarray, y: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and test sets.

        Parameters
        ----------
        X : np.ndarray
            Features matrix.
        y : np.ndarray
            Target vector.

        Returns
        -------
        tuple
            Training and test sets.
        """
        return train_test_split(X, y, **kwargs)

    def normalize_data(self, **kwargs):
        """
        Normalize the data.
        """
        scaler = StandardScaler(**kwargs)
        self.X_train: np.ndarray = scaler.fit_transform(self.X_train)
        self.X_test: np.ndarray = scaler.transform(self.X_test)
        return self

    def evaluate_model(self, model: BaseEstimator, params: Dict[str, Any], **kwargs):
        """
        Evaluate a model.

        Parameters
        ----------
        model : BaseEstimator
            Model to evaluate.
        params : dict
            Parameters to search.
        """
        grid = GridSearchCV(model(), params, scoring=self.scoring, **kwargs)
        grid.fit(self.X_train, self.y_train)
        self.fitted[model.__qualname__] = grid.best_estimator_

        if self.scoring == "neg_mean_squared_error":
            y_pred: np.ndarray = grid.predict(self.X_test)
            score: np.ndarray[Any, np.dtype[Any]] = np.sqrt(mean_squared_error(self.y_test, y_pred))
        else:
            score = grid.best_score_

        if not self.metrics:
            self._scores = {self.scoring: score}
        else:
            self._scores = {
                name: make_scorer(metric)(grid.best_estimator_, self.X_test, self.y_test)
                for name, metric in self.metrics.items()
            }

        return self

    def get_best_model(self) -> Tuple[Any, Dict]:
        """
        Get the best model and its parameters based on the scoring metric.

        This method identifies the best model based on the scoring metric specified during the grid search.
        It first determines the name of the best model by finding the maximum score in the metrics dictionary.
        Then, it sets the best model and its parameters as instance variables.

        Returns
        -------
        tuple
            A tuple containing the best model and its parameters. The first element is the best model
            and the second element is a dictionary of the best parameters for that model.
        """
        best_model_name = max(self._metrics.items(), key=lambda x: x[1][self.scoring])[0]
        self.best_model = [model for model in self.models_params.keys() if model.__qualname__ == best_model_name][0]
        self.best_model = self.fitted[best_model_name]
        self.best_params = self._params[best_model_name]
        return self.best_model, self.best_params

    def search(self, **kwargs):
        """
        Perform grid search for each model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for grid search.

        Returns
        -------
        self : GridSearch
            The fitted GridSearch object.

        Notes
        -----
        This method performs grid search for each model in the `models_params` dictionary. It evaluates each model using
        the specified scoring metric and selects the best model based on the evaluation results.

        Examples
        --------
        >>> search_params = {'param1': [1, 2, 3], 'param2': ['a', 'b', 'c']}
        >>> grid_search = GridSearch(models_params, scoring='accuracy')
        >>> grid_search.search(params=seach_params)
        """
        self._metrics = {}
        self._params = {i[0].__qualname__: i[1] for i in self.models_params.items()}
        for model, params in tqdm(list(self.models_params.items())):
            self.evaluate_model(model, params, **kwargs)
            self._metrics[model.__qualname__] = self._scores
        return self
