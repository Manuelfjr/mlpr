"""
Module for performing grid search on machine learning models.
"""

import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from pyspark.ml import Estimator, Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.sql import DataFrame
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
        if self.scoring in self.metrics:
            scoring = make_scorer(self.metrics[self.scoring])
        else:
            scoring = self.scoring
        grid = GridSearchCV(model(), params, scoring=scoring, **kwargs)
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


class SparkGridSearch:
    """
    A class used to perform grid search for hyperparameter tuning in PySpark.

    ...

    Attributes
    ----------
    models_params : dict
        A dictionary containing the models and their parameters to be used in the grid search.
    features : list
        A list of feature names to be used in the models.
    scoring : str
        The scoring metric to be used for evaluating the models.
    model_type : str
        The type of model ('regression' or 'classification').
    metrics : dict, optional
        A dictionary containing the metrics to be used for evaluating the models (default is None).
    results : dict
        A dictionary to store the results of the grid search.
    best_model : str
        The name of the best model found in the grid search.
    best_cv_model : CrossValidatorModel
        The best CrossValidatorModel found in the grid search.
    best_params : dict
        The parameters of the best model found in the grid search.
    cross_validator_params : dict
        A dictionary containing the parameters to be used in the CrossValidator (default is None).
    numFolds : int
        The number of folds to be used in the CrossValidator (default is 2).

    Methods
    -------
    evaluate_model(df, target, model, params)
        Evaluates a model using cross-validation.
    get_best_model()
        Returns the best model and its parameters.
    add_metric(metric_name, func)
        Adds a new metric to the results.
    predict(df)
        Makes predictions using the best model.
    fit(X, Y)
        Performs the grid search.
    """

    def __init__(
        self,
        models_params: dict[str, dict[str, list[Union[int, float, str]]]],
        features: list[str],
        scoring: str,
        model_type: str,
        metrics: dict[str, dict[str, Union[str, RegressionEvaluator, BinaryClassificationEvaluator]]] = None,
        outputCol: str = "features",
        predictionCol: str = "prediction",
        cross_validator_params: dict = None,
        numFolds: int = 2,
    ):
        """
        Constructs all the necessary attributes for the SparkGridSearch object.

        Parameters
        ----------
            models_params : dict
                A dictionary containing the models and their parameters to be used in the grid search.
            features : list
                A list of feature names to be used in the models.
            scoring : str
                The scoring metric to be used for evaluating the models.
            model_type : str
                The type of model ('regression' or 'classification').
            metrics : dict, optional
                A dictionary containing the metrics to be used for evaluating the models (default is None).
            outputCol : str, optional
                The name of the output column (default is 'prediction').
            predictionCol : str, optional
                The name of the prediction column (default is 'prediction').
            cross_validator_params : dict, optional
                A dictionary containing the parameters to be used in the CrossValidator (default is None).
            numFolds : int, optional
                The number of folds to be used in the CrossValidator (default is 2).
        """
        self.models_params: dict = models_params
        self.features: list[str] = features
        self.scoring: str = scoring
        self.model_type: str = model_type
        self.outputCol: str = outputCol
        self.predictionCol: str = predictionCol
        self.numFolds = numFolds
        if self.model_type == "regression":
            self.metrics: dict[str, dict] = (
                metrics
                if metrics
                else {"rmse": {"evaluator": RegressionEvaluator(metricName="rmse"), "goal": "minimize"}}
            )
        elif self.model_type == "classification":
            self.metrics: dict[str, dict] = (
                metrics
                if metrics
                else {
                    "areaUnderROC": {
                        "evaluator": BinaryClassificationEvaluator(metricName="areaUnderROC"),
                        "goal": "maximize",
                    }
                }
            )
        else:
            raise ValueError("Invalid model_type. Expected 'regression' or 'classification'")
        self.results = {}
        self.best_model = None
        self.best_cv_model = None
        self.best_params = None
        self.cross_validator_params = cross_validator_params if cross_validator_params else {}

    def evaluate_model(
        self, df: DataFrame, target: str, model: Estimator, params: dict[str, list[Union[int, float, str]]]
    ) -> CrossValidatorModel:
        """
        Evaluates a model using cross-validation.

        Parameters
        ----------
            df : DataFrame
                The data to be used for training and evaluating the model.
            target : str
                The target variable.
            model : Estimator
                The model to be evaluated.
            params : dict
                The parameters to be used in the model.

        Returns
        -------
            CrossValidatorModel
                The trained CrossValidatorModel.
        """
        assembler = VectorAssembler(inputCols=self.features, outputCol=self.outputCol)
        model.setLabelCol(target)

        paramGridBuilder = ParamGridBuilder()

        for param, values in params.items():
            paramGridBuilder = paramGridBuilder.addGrid(getattr(model, param), values)

        model_metrics = {}
        for metric_name, metric_info in self.metrics.items():
            evaluator = metric_info["evaluator"]
            evaluator.setLabelCol(target)
            evaluator.setPredictionCol(self.predictionCol)
            crossval = CrossValidator(
                estimator=Pipeline(stages=[assembler, model]),
                estimatorParamMaps=paramGridBuilder.build(),
                evaluator=evaluator,
                numFolds=self.numFolds,
                **self.cross_validator_params,
            )
            cvModel = crossval.fit(df)
            model_metrics[metric_name] = cvModel.avgMetrics
            model_metrics["params"] = [dict(zip(params.keys(), x)) for x in cvModel.getEstimatorParamMaps()]
        self.results[model.__class__.__name__] = model_metrics
        return cvModel

    def get_best_model(self) -> Tuple[Union[CrossValidatorModel, None], Dict]:
        """
        Returns the best model and its parameters.

        Returns
        -------
            Tuple[Union[CrossValidatorModel, None], dict]
                The best model and its parameters.
        """
        return self.best_cv_model, self.best_params

    def add_metric(self, metric_name: str, func: callable) -> "SparkGridSearch":
        """
        Adds a new metric to the results.

        Parameters
        ----------
            metric_name : str
                The name of the metric.
            func : callable
                The function to calculate the metric.

        Returns
        -------
            SparkGridSearch
                The current SparkGridSearch object.
        """
        for _, model_metrics in self.results.items():
            if metric_name in model_metrics:
                model_metrics[metric_name] = list(map(func, model_metrics[metric_name]))
        return self

    def predict(self, df: DataFrame) -> DataFrame:
        """
        Makes predictions using the best model.

        Parameters
        ----------
            df : DataFrame
                The data to be used for making predictions.

        Returns
        -------
            DataFrame
                The predictions made by the best model.
        """
        if self.best_cv_model is None:
            raise RuntimeError("No model has been trained yet.")

        predictions = self.best_cv_model.transform(df)
        return predictions

    def update_best_model(self, cvModel: CrossValidatorModel, model_class: type[Estimator]):
        """
        Updates the best model if the current model is better.

        Parameters
        ----------
            cvModel : CrossValidatorModel
                The current model.
            model_class : type[Estimator]
                The class of the current model.
        """
        current_model_score = max(self.results[model_class.__name__][self.scoring])
        best_model_score = max(self.results[self.best_model][self.scoring]) if self.best_model else float("-inf")

        is_maximize_goal = self.metrics[self.scoring]["goal"] == "maximize"
        is_minimize_goal = self.metrics[self.scoring]["goal"] == "minimize"
        is_best_model_none = self.best_model is None
        is_current_better = current_model_score > best_model_score if best_model_score is not None else True
        is_current_worse = current_model_score < best_model_score if best_model_score is not None else False

        if is_best_model_none or (is_maximize_goal and is_current_better) or (is_minimize_goal and is_current_worse):
            self.set_best_model(cvModel, model_class)

        return self

    def set_best_model(self, cvModel: CrossValidatorModel, model_class: type[Estimator]):
        """
        Sets the current model as the best model.

        Parameters
        ----------
            cvModel : CrossValidatorModel
                The current model.
            model_class : type[Estimator]
                The class of the current model.
        """
        self.best_model = model_class.__name__
        self.best_cv_model = cvModel

        model_name = model_class.__name__
        model_results = self.results[model_name]
        params_list = model_results["params"]
        best_params_index = cvModel.avgMetrics.index(max(cvModel.avgMetrics))
        self.best_params = params_list[best_params_index]
        return self

    def fit(self, X: DataFrame, Y: str):
        """
        Performs the grid search.

        Parameters
        ----------
            X : DataFrame
                The features to be used for training the models.
            Y : str
                The target variable.

        Returns
        -------
            SparkGridSearch
                The current SparkGridSearch object.
        """
        for model_class, params in self.models_params.items():
            cvModel = self.evaluate_model(X, Y, model_class(), params)
            self.update_best_model(cvModel, model_class)
        return self
