"""
Metrics module for supervisioned learning.
"""

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyspark.sql
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn import metrics as skmetrics


class ClassificationMetrics:
    """
    Compute classification metrics for pandas and PySpark dataframes.

    Parameters
    ----------
    metrics : list of str or callable, default=None
        The metrics to be computed. If None, all metrics are computed.
        If a list of str, the strings should be valid sklearn.metrics or
        pyspark.ml.evaluation function names. If a callable, it should be
        a function that takes two arguments (y_true and y_pred) and returns
        a scalar.

    Attributes
    ----------
    results_ : dict
        A dictionary where the keys are the names of the metrics and the
        values are the computed metrics.
    """

    def __init__(
        self,
        metrics: Optional[Union[List[str], Callable]] = None,
        true_target: Optional[str] = "y_true",
        pred_target: Optional[str] = "y_pred",
    ):
        self.metrics = metrics
        self.results_: Dict[str, float] = {}
        self.true_target: str = true_target
        self.pred_target: str = pred_target

    def add_metric(
        self,
        name: str,
        metric_fn: Callable[
            [
                Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
                Union[pd.DataFrame, pyspark.sql.DataFrame],
            ],
            float,
        ],
    ) -> "ClassificationMetrics":
        """
        Add a custom metric to the metrics dictionary.

        Parameters
        ----------
        name : str
            The name of the metric.
        metric_fn : callable
            The metric function. It should take two arguments (y_true and y_pred) and return a scalar.
        """
        if not callable(metric_fn):
            raise ValueError("metric_fn should be a callable function")
        setattr(self, name, metric_fn)
        return self

    def accuracy_score(
        self,
        y_true: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
        **kwargs,
    ) -> float:
        """
        Compute the accuracy score.

        Parameters
        ----------
        y_true : pandas.DataFrame or pyspark.sql.DataFrame
            The ground truth labels.
        y_pred : pandas.DataFrame or pyspark.sql.DataFrame
            The predicted labels.

        Returns
        -------
        float
            The computed accuracy score.
        """
        is_dataframe: bool = isinstance(y_true, pd.DataFrame) and isinstance(y_pred, pd.DataFrame)
        is_series: bool = isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series)
        is_ndarray: bool = isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)

        if is_dataframe or is_series or is_ndarray:
            return skmetrics.accuracy_score(y_true, y_pred, **kwargs)
        if isinstance(y_true, pyspark.sql.DataFrame) and isinstance(y_pred, pyspark.sql.DataFrame):
            evaluator = MulticlassClassificationEvaluator(
                labelCol=self.true_target, predictionCol=self.pred_target, metricName="accuracy", **kwargs
            )
            return evaluator.evaluate(y_pred)
        raise ValueError(
            """
            y_true and y_pred arguments are not supported.
            Must be of type:
                pandas.DataFrame
                pandas.Series
                numpy.ndarray
                pyspark.sql.DataFrame
            """
        )

    def precision_score(
        self,
        y_true: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
        **kwargs,
    ) -> float:
        """
        Compute the precision score.

        Parameters
        ----------
        y_true : pandas.DataFrame or pyspark.sql.DataFrame
            The ground truth labels.
        y_pred : pandas.DataFrame or pyspark.sql.DataFrame
            The predicted labels.

        Returns
        -------
        float
            The computed precision score.
        """
        is_dataframe = isinstance(y_true, pd.DataFrame) and isinstance(y_pred, pd.DataFrame)
        is_series = isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series)
        is_ndarray = isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)

        if is_dataframe or is_series or is_ndarray:
            return skmetrics.precision_score(y_true, y_pred, **kwargs)
        if isinstance(y_true, pyspark.sql.DataFrame) and isinstance(y_pred, pyspark.sql.DataFrame):
            evaluator = MulticlassClassificationEvaluator(
                labelCol=self.true_target, predictionCol=self.pred_target, metricName="weightedPrecision", **kwargs
            )
            return evaluator.evaluate(y_pred)
        raise ValueError(
            """
            y_true and y_pred arguments are not supported.
            Must be of type:
                pandas.DataFrame
                pandas.Series
                numpy.ndarray
                pyspark.sql.DataFrame
            """
        )

    def recall_score(
        self,
        y_true: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
        **kwargs,
    ) -> float:
        """
        Compute the recall score.

        Parameters
        ----------
        y_true : pandas.DataFrame or pyspark.sql.DataFrame
            The ground truth labels.
        y_pred : pandas.DataFrame or pyspark.sql.DataFrame
            The predicted labels.

        Returns
        -------
        float
            The computed recall score.
        """
        is_dataframe = isinstance(y_true, pd.DataFrame) and isinstance(y_pred, pd.DataFrame)
        is_series = isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series)
        is_ndarray = isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)

        if is_dataframe or is_series or is_ndarray:
            return skmetrics.recall_score(y_true, y_pred, **kwargs)
        if isinstance(y_true, pyspark.sql.DataFrame) and isinstance(y_pred, pyspark.sql.DataFrame):
            evaluator = MulticlassClassificationEvaluator(
                labelCol=self.true_target, predictionCol=self.pred_target, metricName="weightedRecall", **kwargs
            )
            return evaluator.evaluate(y_pred)
        raise ValueError(
            """
            y_true and y_pred arguments are not supported.
            Must be of type:
                pandas.DataFrame
                pandas.Series
                numpy.ndarray
                pyspark.sql.DataFrame
            """
        )

    def f1_score(
        self,
        y_true: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
        **kwargs,
    ) -> float:
        """
        Compute the F1 score.

        Parameters
        ----------
        y_true : pandas.DataFrame or pyspark.sql.DataFrame
            The ground truth labels.
        y_pred : pandas.DataFrame or pyspark.sql.DataFrame
            The predicted labels.

        Returns
        -------
        float
            The computed F1 score.
        """
        is_dataframe = isinstance(y_true, pd.DataFrame) and isinstance(y_pred, pd.DataFrame)
        is_series = isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series)
        is_ndarray = isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)

        if is_dataframe or is_series or is_ndarray:
            return skmetrics.f1_score(y_true, y_pred, **kwargs)
        if isinstance(y_true, pyspark.sql.DataFrame) and isinstance(y_pred, pyspark.sql.DataFrame):
            evaluator = MulticlassClassificationEvaluator(
                labelCol=self.true_target, predictionCol=self.pred_target, metricName="f1", **kwargs
            )
            return evaluator.evaluate(y_pred)
        raise ValueError(
            """
            y_true and y_pred arguments are not supported.
            Must be of type:
                pandas.DataFrame
                pandas.Series
                numpy.ndarray
                pyspark.sql.DataFrame
            """
        )

    def roc_auc_score(
        self,
        y_true: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
        **kwargs,
    ) -> float:
        """
        Compute the ROC AUC score.

        Parameters
        ----------
        y_true : pandas.DataFrame or pyspark.sql.DataFrame
            The ground truth labels.
        y_pred : pandas.DataFrame or pyspark.sql.DataFrame
            The predicted labels.

        Returns
        -------
        float
            The computed ROC AUC score.
        """
        is_dataframe = isinstance(y_true, pd.DataFrame) and isinstance(y_pred, pd.DataFrame)
        is_series = isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series)
        is_ndarray = isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)

        if is_dataframe or is_series or is_ndarray:
            return skmetrics.roc_auc_score(y_true, y_pred, **kwargs)
        if isinstance(y_true, pyspark.sql.DataFrame) and isinstance(y_pred, pyspark.sql.DataFrame):
            evaluator = BinaryClassificationEvaluator(
                labelCol=self.true_target, rawPredictionCol=self.pred_target, metricName="areaUnderROC", **kwargs
            )
            return evaluator.evaluate(y_pred)
        raise ValueError(
            """
            y_true and y_pred arguments are not supported.
            Must be of type:
                pandas.DataFrame
                pandas.Series
                numpy.ndarray
                pyspark.sql.DataFrame
            """
        )

    def confusion_matrix(
        self,
        y_true: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
        **kwargs,
    ) -> np.ndarray:
        """
        Compute the confusion matrix.

        Parameters
        ----------
        y_true : pandas.DataFrame or pyspark.sql.DataFrame
            The ground truth labels.
        y_pred : pandas.DataFrame or pyspark.sql.DataFrame
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The confusion matrix.
        """
        is_dataframe = isinstance(y_true, pd.DataFrame) and isinstance(y_pred, pd.DataFrame)
        is_series = isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series)
        is_ndarray = isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)

        if is_dataframe or is_series or is_ndarray:
            return skmetrics.confusion_matrix(y_true, y_pred, **kwargs)
        if isinstance(y_true, pyspark.sql.DataFrame) and isinstance(y_pred, pyspark.sql.DataFrame):
            predictionAndLabels = (
                y_pred.select(self.pred_target)
                .rdd.map(lambda x: float(x[0]))
                .zip(y_true.select(self.true_target).rdd.map(lambda x: float(x[0])))
            )
            metrics = MulticlassMetrics(predictionAndLabels)
            confusion_matrix = metrics.confusionMatrix().toArray()
            return confusion_matrix
        raise ValueError(
            """
            y_true and y_pred arguments are not supported.
            Must be of type:
                pandas.DataFrame
                pandas.Series
                numpy.ndarray
                pyspark.sql.DataFrame
            """
        )

    def compute_metrics(
        self,
        y_true: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
        y_pred: Union[pd.DataFrame, pyspark.sql.DataFrame, pd.Series, np.ndarray],
    ) -> "ClassificationMetrics":
        """
        Compute all the specified classification metrics.

        Parameters
        ----------
        y_true : pandas.DataFrame or pyspark.sql.DataFrame
            The ground truth labels.
        y_pred : pandas.DataFrame or pyspark.sql.DataFrame
            The predicted labels.

        Returns
        -------
        self
            The ClassificationMetrics object with the computed metrics.
        """
        if self.metrics is None:
            self.metrics: dict[str, dict] = {
                "accuracy_score": {},
                "precision_score": {},
                "recall_score": {},
                "f1_score": {},
                "roc_auc_score": {},
            }

        for metric, params in self.metrics.items():
            if callable(metric):
                self.results_[metric.__name__] = metric(y_true, y_pred, **params)
                if isinstance(self.results_[metric.__name__], (float, int)):
                    self.results_[metric.__name__] = [self.results_[metric.__name__]]
            elif isinstance(metric, str):
                if hasattr(self, metric):
                    self.results_[metric] = getattr(self, metric)(y_true, y_pred, **params)
                    if isinstance(self.results_[metric], (float, int)):
                        self.results_[metric] = [self.results_[metric]]
                else:
                    raise ValueError(f"Invalid metric: {metric}")
            else:
                raise ValueError("Metrics should be either callable or a dictionary of str and parameters")
        return self
