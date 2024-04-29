from typing import Any, Dict, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, get_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

class GridSearch:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models_params: Dict[BaseEstimator, Dict[str, Any]],
        params_split: dict = {},
        normalize: bool = True,
        params_norm: dict = {},
        scoring: str = 'neg_mean_squared_error'
    ):
        """
        Initialize GridSearch object.

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
        scoring : str, default='neg_mean_squared_error'
            Scoring metric to evaluate the models. Must be a valid scoring metric for sklearn's GridSearchCV.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(X, y, **params_split)
        self.models_params = models_params

        if normalize:
            self.normalize_data(**params_norm)

        self.best_score_ = None
        self.best_model = None
        self.best_params = None
        self.scoring = scoring
        self.best_score_ = np.inf if get_scorer(scoring)._sign == 1 else -np.inf

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
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
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

        if self.scoring == 'neg_mean_squared_error':
            y_pred = grid.predict(self.X_test)
            score = np.sqrt(mean_squared_error(self.y_test, y_pred))
        else:
            score = grid.best_score_

        if (get_scorer(self.scoring)._sign == 1 and score < self.best_score_) or (get_scorer(self.scoring)._sign == -1 and score > self.best_score_):
            self.best_score_ = score
            self.best_model = grid.best_estimator_  # store the trained model
            self.best_params = grid.best_params_

        return self

    def get_best_model(self) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Get the best model and its parameters.

        Returns
        -------
        tuple
            Best model and its parameters.
        """
        return self.best_model, self.best_params

    def search(self, **kwargs):
        """
        Perform grid search for each model.
        """
        for model, params in self.models_params.items():
            self.evaluate_model(model, params, **kwargs)
        return self
