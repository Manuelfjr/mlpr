"""
This module provides a class for evaluating regression models and calculating various metrics.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import KBinsDiscretizer


# pylint: disable=too-many-instance-attributes
class RegressionMetrics:
    """
    A class for evaluating regression models.

    Parameters
    ----------
    data : Union[pd.DataFrame, DataFrame]
        The data used for evaluation.
    target_col : str, default="y_true"
        The name of the column in data that contains the true target values.
    preds_col : str, default="y_pred"
        The name of the column in data that contains the predicted target values.

    Attributes
    ----------
    y_true : Series
        The true target values.
    y_pred : Series
        The predicted target values.
    """

    def __init__(
        self, data: Union[pd.DataFrame, DataFrame], target_col: str = "y_true", preds_col: str = "y_pred"
    ) -> None:
        self.data: pd.DataFrame | DataFrame = data
        self.target_col: str = target_col
        self.preds_col: str = preds_col
        self.y_true: pd.Series | Column = self.data[self.target_col]
        self.y_pred: pd.Series | Column = self.data[self.preds_col]
        self.cm = None
        self.metrics = None
        self._worst_interval_cm = None
        self._worst_interval_kappa = None
        self._metrics_cm = None
        self._class_intervals = None

    def __search(
        self,
        metrics_dict: dict = None,
        intervals_dict: dict = None,
        metric: str = "precision",
        minimize: Union[int, bool] = 1,
        use: str = "kappa",
    ):
        if metrics_dict is None:
            metrics_dict = {}
        if intervals_dict is None:
            intervals_dict = {}
        if use == "kappa":
            method: dict = {
                i_class: (
                    content["metrics"][metric][0]
                    if len(content["metrics"][metric]) == 1
                    else content["metrics"][metric][int(minimize)]
                )
                for i_class, content in metrics_dict.items()
            }
            worst_class: int | str = min(method, key=method.get)
        elif use == "confusion_matrix":
            worst_class: int | str = np.where(
                metrics_dict["metrics"][metric] == np.min(metrics_dict["metrics"][metric])
            )[0][0]
        return intervals_dict[worst_class]

    def _discretize_data(
        self,
        data: pd.DataFrame,
        n_bins: int,
        encode: str = "ordinal",
        strategy: str = "uniform",
        subsample: int = 200000,
        **kwargs,
    ) -> tuple[pd.Series, pd.Series, KBinsDiscretizer]:
        """
        Discretize the data into bins.

        Parameters
        ----------
        data : pd.DataFrame
            The data to discretize.
        n_bins : int
            The number of bins to use.
        encode : str, default="ordinal"
            The method used to encode the transformed result.
        strategy : str, default="uniform"
            The strategy used to define the widths of the bins.
        subsample : int, default=200000
            The maximum number of samples used to estimate the quantiles for subsampling.
        **kwargs
            Additional parameters to pass to the discretization function.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, KBinsDiscretizer]
            A tuple containing the true bins, predicted bins, and the discretizer object.

        Notes
        -----
        This method discretizes the target column and the predicted column of the input data into bins.
        It uses the `KBinsDiscretizer` class from scikit-learn to perform the discretization.
        The true bins and predicted bins are returned as numpy arrays, and the discretizer object is also returned.

        Example
        -------
        >>> data = pd.DataFrame({'target': [1.2, 2.5, 3.7, 4.1, 5.0], 'preds': [1.0, 2.2, 3.8, 4.3, 5.2]})
        >>> n_bins = 3
        >>> encode = 'ordinal'
        >>> strategy = 'uniform'
        >>> subsample = 200000
        >>> true_bins, pred_bins, discretizer = _discretize_data(data, n_bins, encode, strategy, subsample)
        """
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy, subsample=subsample, **kwargs)
        discretizer.fit(data[["y_true"]].values)

        data["true_bins"] = discretizer.transform(data[["y_true"]].values)
        data["pred_bins"] = discretizer.transform(data[["y_pred"]].values)
        return data["true_bins"], data["pred_bins"], discretizer

    def _calculate_mape(self) -> float:
        """
        Calculate the Mean Absolute Percentage Error (MAPE).

        Returns
        -------
        float
            The MAPE of the predictions.
        """
        absolute_percentage_error: float = np.abs((self.y_true - self.y_pred) / self.y_true)
        mape = np.mean(absolute_percentage_error) * 100
        return mape

    def _calculate_spark_mape(self) -> float:
        """
        Calculate the Mean Absolute Percentage Error (MAPE) using Spark.

        Returns
        -------
        float
            The MAPE of the predictions.
        """
        true_pred_df: DataFrame = self.data.select(self.target_col, self.preds_col)
        true_pred_df: DataFrame = true_pred_df.withColumn(
            "absolute_percentage_error",
            (F.abs(F.col(self.target_col) - F.col(self.preds_col)) / F.col(self.target_col)),
        )
        mape = true_pred_df.select(F.mean("absolute_percentage_error")).first()[0] * 100
        return mape

    def _calculate_rmse(self, **kwargs) -> np.ndarray:
        """
        Calculate the Root Mean Square Error (RMSE).

        **kwargs
            Additional parameters to pass to the RMSE calculation function.

        Returns
        -------
        float
            The RMSE of the predictions.
        """
        return np.sqrt(mean_squared_error(self.y_true, self.y_pred, **kwargs))

    def _calculate_spark_rmse(self) -> float:
        """
        Calculate the Root Mean Square Error (RMSE) using Spark.

        Returns
        -------
        float
            The RMSE of the predictions.
        """
        true_pred_df: DataFrame = self.data.select(self.target_col, self.preds_col)
        true_pred_df: DataFrame = true_pred_df.withColumn(
            "squared_error", F.pow(F.col(self.target_col) - F.col(self.preds_col), 2)
        )
        mse: float = true_pred_df.select(F.mean("squared_error")).first()[0]
        rmse: float = np.sqrt(mse)
        return rmse

    def _calculate_ks(self, data: pd.DataFrame, **kwargs) -> tuple[float, float]:
        """
        Calculate the Kolmogorov-Smirnov statistic on 2 samples.

        Parameters
        ----------
        data
            Input data.
        **kwargs
            Additional parameters to pass to the ks_2samp function.

        Returns
        -------
        tuple[float, float]
            KS statistic and two-tailed p-value.
        """

        y_true: pd.Series = data[self.target_col]
        y_pred: pd.Series = data[self.preds_col]
        ks_statistic, p_value = ks_2samp(y_true, y_pred, **kwargs)
        return ks_statistic, p_value

    def _calculate_spark_ks(self) -> tuple[float, float]:
        """
        Calculate the Kolmogorov-Smirnov statistic on 2 samples using Spark.

        Returns
        -------
        tuple[float, float]
            KS statistic and two-tailed p-value.
        """
        pandas_df: pd.DataFrame = self.data.toPandas()
        return self._calculate_ks(pandas_df)

    def _get_interval_class(self, n_bins: int, cutoff_bins: np.ndarray) -> dict:
        """
        Get the intervals for each class after discretizing the true and predicted values into bins.

        Parameters
        ----------
        n_bins : int
            The number of bins to discretize the true and predicted values into.
        cutoff_bins: np.ndarray
            The array of cutoff points for the bins.

        Returns
        -------
        dict
            A dictionary where the keys are the classes (or bins) and the values are the corresponding intervals.
        """
        self._class_intervals: dict = {}
        for i in range(n_bins):
            if i == 0:
                self._class_intervals[i] = (-float("inf"), cutoff_bins[i + 1])
            elif i == n_bins - 1:
                self._class_intervals[i] = (cutoff_bins[i], float("inf"))
            else:
                self._class_intervals[i] = (cutoff_bins[i], cutoff_bins[i + 1])
        return self._class_intervals

    def _get_metrics_cm(self, true_bins: np.ndarray, pred_bins: np.ndarray) -> dict:
        """
        Calculate various metrics from a confusion matrix.

        Parameters
        ----------
        true_bins : np.ndarray
            The true target values discretized into bins.
        pred_bins : np.ndarray
            The predicted target values discretized into bins.

        Returns
        -------
        dict
            A dictionary where the keys are the metric names and the values are the corresponding metric values.
        """
        precision, recall, f1_score, support = precision_recall_fscore_support(
            true_bins, pred_bins, labels=np.unique(true_bins)
        )
        accuracy: float = accuracy_score(true_bins, pred_bins)
        metrics: dict[str, any] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support,
            "accuracy": accuracy,
        }
        return metrics

    def mape(self) -> float:
        """
        Public method to calculate the Mean Absolute Percentage Error (MAPE).

        Returns
        -------
        float
            The MAPE of the predictions.
        """
        return self._calculate_mape() if isinstance(self.data, pd.DataFrame) else self._calculate_spark_mape()

    def rmse(self, **kwargs) -> float:
        """
        Calculate the Root Mean Square Error (RMSE).

        **kwargs
            Additional parameters to pass to the RMSE calculation function.

        Returns
        -------
        float
            The RMSE of the predictions.
        """
        return self._calculate_rmse(**kwargs) if isinstance(self.data, pd.DataFrame) else self._calculate_spark_rmse()

    def kolmogorov_smirnov(self, **kwargs) -> tuple[float, float]:
        """
        Calculate the Kolmogorov-Smirnov statistic on 2 samples.

        **kwargs
            Additional parameters to pass to the ks_2samp function.

        Returns
        -------
        tuple[float, float]
            KS statistic and two-tailed p-value.
        """
        return (
            self._calculate_ks(self.data, **kwargs)
            if isinstance(self.data, pd.DataFrame)
            else self._calculate_spark_ks()
        )

    def confusion_matrix(
        self,
        n_bins: int,
        encode: str = "ordinal",
        strategy: str = "uniform",
        metric: str = "precision",
        minimize: Union[int, bool] = True,
        **kwargs,
    ) -> dict:
        """
        Get the confusion matrix and metrics for each class after discretizing the true and predicted values into bins.

        Parameters
        ----------
        n_bins : int
            The number of bins to discretize the true and predicted values into.
        encode : str, default="ordinal"
            The encoding method for the bins.
        strategy : str, default="uniform"
            The strategy used to define the widths of the bins.
        metric : str, default="precision"
            The metric used for searching the worst interval.
        minimize : Union[int, bool], default=True
            Determines whether to minimize or maximize the metric.
        **kwargs : dict
            Additional keyword arguments to be passed to the discretizer.

        Returns
        -------
        dict
            A dictionary where the keys are the classes (or bins) and the values are tuples containing the
            corresponding confusion matrix and metrics.
        """
        true_bins, pred_bins, discretizer = self._discretize_data(
            self.data, n_bins=n_bins, encode=encode, strategy=strategy, **kwargs
        )
        kappa: float = cohen_kappa_score(true_bins, pred_bins)

        self._get_interval_class(n_bins, discretizer.bin_edges_[0])
        self.cm: np.ndarray = confusion_matrix(true_bins, pred_bins)
        self._metrics_cm: dict = self._get_metrics_cm(true_bins, pred_bins)
        results: dict = {
            "confusion_matrix": self.cm,
            "kappa_score": kappa,
            "metrics": self._metrics_cm,
        }
        self._worst_interval_cm: tuple = self.__search(
            results, self._class_intervals, metric=metric, minimize=minimize, use="confusion_matrix"
        )
        return self.cm, self._metrics_cm

    def calculate_kappa(  # pylint: disable=too-many-locals
        self,
        n_bins: int,
        encode: str = "ordinal",
        strategy: str = "uniform",
        metric: str = "precision",
        minimize: Union[int, bool] = True,
        **kwargs,
    ) -> dict:
        """
        Calculate a binary confusion matrix and Cohen's Kappa for each class.

        Parameters
        ----------
        n_bins : int
            The number of bins to discretize the true and predicted values into.
        encode : str, default="ordinal"
            The method used to encode the bins. Options are "ordinal", "onehot", "onehot-dense", "ordinal".
        strategy : str, default="uniform"
            The strategy used to define the widths of the bins. Options are "uniform", "quantile", "kmeans".
        metric : str, default="precision"
            The metric used for searching the worst interval.
        minimize : Union[int, bool], default=True
            Determines whether to minimize or maximize the metric.
        **kwargs : dict
            Additional keyword arguments to be passed to the discretizer.

        Returns
        -------
        dict
            A dictionary where the keys are the classes (or bins) and the values are dictionaries containing
            the corresponding binary confusion matrix, Cohen's Kappa, and other metrics.
        """
        true_bins, pred_bins, discretizer = self._discretize_data(
            self.data, n_bins=n_bins, encode=encode, strategy=strategy, **kwargs
        )
        self._get_interval_class(n_bins, discretizer.bin_edges_[0])

        binary_cms_and_kappas: dict = {}
        for i in range(n_bins):
            true_bins_binary: pd.Series = (true_bins == i).astype(int)
            pred_bins_binary: pd.Series = (pred_bins == i).astype(int)
            cm: np.ndarray = confusion_matrix(true_bins_binary, pred_bins_binary)
            kappa: float = cohen_kappa_score(true_bins_binary, pred_bins_binary)
            binary_cms_and_kappas[i] = {
                "confusion_matrix": cm,
                "kappa_score": kappa,
                "metrics": self._get_metrics_cm(true_bins_binary, pred_bins_binary),
            }
        self._worst_interval_kappa: tuple[int | float, int | float] = self.__search(
            binary_cms_and_kappas, self._class_intervals, metric=metric, minimize=minimize
        )
        return binary_cms_and_kappas

    def calculate_metrics(self, metrics_list: list = None, metrics_params: dict = None, **kwargs) -> dict:
        """
        Calculates the metrics specified in the "metrics_list" parameter.

        Parameters
        ----------
        metrics_list : list
            A list of metric function names to be calculated.
        metrics_params : dict
            A dictionary containing additional parameters for each metric function.
        **kwargs
            Additional keyword arguments that can be passed to the metric functions.

        Returns
        -------
        dict
            A dictionary containing the calculated metrics, where the keys are the metric function names
            and the values are the calculated metric values.
        """
        if metrics_list is None:
            metrics_list = []
        if metrics_params is None:
            metrics_params = {}

        self.metrics: dict = {}
        for func_name in metrics_list:
            func: any = getattr(self, func_name)
            args: dict[str, any] = {**kwargs, **metrics_params.get(func_name, {})}
            self.metrics[func_name] = func(**args)
        return self.metrics

    def spark_custom_metrics(
        self,
        predictions: DataFrame,
        labelCol: str = "label",
        predictionCol: str = "prediction",
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate a regression model using the specified metrics.

        Parameters
        ----------
        predictions : pyspark.sql.DataFrame
            A DataFrame containing the predictions and true labels.
        labelCol : str, default="label"
            The name of the column containing the true labels.
        predictionCol : str, default="prediction"
            The name of the column containing the predictions.
        metrics : list, default=["mse", "rmse", "r2", "mae"]
            A list of metrics to calculate.
        **kwargs
            Additional keyword arguments to pass to the RegressionEvaluator.

        Returns
        -------
        dict
            A dictionary where the keys are the metric names and the values are the calculated metric values.
        """
        if metrics is None:
            metrics = ["mse", "rmse", "r2", "mae"]

        # pylint: disable=import-outside-toplevel
        from pyspark.ml.evaluation import RegressionEvaluator

        results: dict = {}
        for metric in metrics:
            evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol=predictionCol, metricName=metric, **kwargs)
            value: float = evaluator.evaluate(predictions)
            results[metric] = value
        return results
