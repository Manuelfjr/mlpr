from typing import Dict, Tuple, Any
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

class GridSearch:
    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            models_params: Dict[BaseEstimator, Dict[str, Any]],
            params_split: dict={},
            normalize: bool = True,
            params_norm: dict={}
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
        test_size : float, default=0.2
            Test set size.
        random_state : int, default=None
            Random state for train-test split.
        normalize : bool, default=True
            Whether to normalize the data.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(X, y, **params_split)
        self.models_params = models_params

        if normalize:
            self.normalize_data(**params_norm)

        self.best_score_ = None
        self.best_model = None
        self.best_params = None
        self.lowest_rmse = np.inf

    def split_data(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        grid = GridSearchCV(model(), params, **kwargs)
        grid.fit(self.X_train, self.y_train)

        y_pred = grid.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

        if rmse < self.lowest_rmse:
            self.lowest_rmse = rmse
            self.best_model = grid.best_estimator_  # store the trained model
            self.best_params = grid.best_params_
            self.best_score_ = grid.best_score_
            
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
    
    def search(self):
        """
        Perform grid search for each model.
        """
        for model, params in self.models_params.items():
            self.evaluate_model(model, params)
        return self