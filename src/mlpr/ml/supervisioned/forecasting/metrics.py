from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pyspark.sql as sparksql
from pyspark.sql import DataFrame
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


class ForecastMetrics:
    """
    A class for evaluating Forecast models.

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

    def __init__(self, data: Union[pd.DataFrame, DataFrame], target_col: str = "y_true", preds_col: str = "y_pred"):
        self.data = data
        self.target_col = target_col
        self.preds_col = preds_col
        self.y_true = self.data[self.target_col]
        self.y_pred = self.data[self.preds_col]

    def _discretize_data(
        self,
        data: pd.DataFrame,
        n_bins: int,
        encode: str = "ordinal",
        strategy: str = "uniform",
        subsample: int = 200000,
        **kwargs,
    ):
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
        absolute_percentage_error = np.abs((self.y_true - self.y_pred) / self.y_true)
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
        true_pred_df = self.data.select(self.target_col, self.preds_col)
        true_pred_df = true_pred_df.withColumn(
            "absolute_percentage_error",
            (F.abs(F.col(self.target_col) - F.col(self.preds_col)) / F.col(self.target_col)),
        )
        mape = true_pred_df.select(F.mean("absolute_percentage_error")).first()[0] * 100
        return mape

    def _calculate_rmse(self, **kwargs) -> float:
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
        true_pred_df = self.data.select(self.target_col, self.preds_col)
        true_pred_df = true_pred_df.withColumn(
            "squared_error", F.pow(F.col(self.target_col) - F.col(self.preds_col), 2)
        )
        mse = true_pred_df.select(F.mean("squared_error")).first()[0]
        rmse = np.sqrt(mse)
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

        y_true = data[self.target_col]
        y_pred = data[self.preds_col]
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
        pandas_df = self.data.toPandas()
        return self._calculate_ks(pandas_df)

    def _get_interval_class(self, n_bins: int, cutoff_bins: np.ndarray):
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
        self._class_intervals = {}
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
        accuracy = accuracy_score(true_bins, pred_bins)
        metrics = {
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

    def confusion_matrix(self, n_bins: int, encode: str = "ordinal", strategy: str = "uniform", **kwargs) -> dict:
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
        class_intervals = {}
        for i in range(n_bins):
            if i == 0:
                class_intervals[i] = (-float("inf"), discretizer.bin_edges_[0][i + 1])
            elif i == n_bins - 1:
                class_intervals[i] = (discretizer.bin_edges_[0][i], float("inf"))
            else:
                class_intervals[i] = (discretizer.bin_edges_[0][i], discretizer.bin_edges_[0][i + 1])
        self.cm = confusion_matrix(true_bins, pred_bins)
        self._metrics_cm = self._get_metrics_cm(true_bins, pred_bins)
        self._get_interval_class(n_bins, discretizer.bin_edges_[0])
        return self.cm, self._metrics_cm

    def calculate_kappa(self, n_bins: int, encode: str = "ordinal", strategy: str = "uniform", **kwargs) -> dict:
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

        Returns
        -------
        dict
            A dictionary where the keys are the classes (or bins) and the values are dictionaries containing
            the corresponding binary confusion matrix, Cohen's Kappa, and other metrics.
        """
        true_bins, pred_bins, _ = self._discretize_data(
            self.data, n_bins=n_bins, encode=encode, strategy=strategy, **kwargs
        )
        binary_cms_and_kappas = {}
        for i in range(n_bins):
            true_bins_binary = (true_bins == i).astype(int)
            pred_bins_binary = (pred_bins == i).astype(int)
            cm = confusion_matrix(true_bins_binary, pred_bins_binary)
            kappa = cohen_kappa_score(true_bins_binary, pred_bins_binary)
            binary_cms_and_kappas[i] = {
                "confusion_matrix": cm,
                "kappa_score": kappa,
                "metrics": self._get_metrics_cm(true_bins_binary, pred_bins_binary),
            }
        return binary_cms_and_kappas

    def calculate_metrics(self, metrics_list: list = [], metrics_params: dict = {}, **kwargs):
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
        self.metrics = {}
        for func_name in metrics_list:
            func = getattr(self, func_name)
            args = {**kwargs, **metrics_params.get(func_name, {})}
            self.metrics[func_name] = func(**args)
        return self.metrics

    def spark_custom_metrics(
        self,
        predictions: sparksql.DataFrame,
        labelCol: str = "label",
        predictionCol: str = "prediction",
        metrics: List[str] = ["mse", "rmse", "r2", "mae"],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate a Forecast model using the specified metrics.

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
            Additional keyword arguments to pass to the RegressionEvaluation.

        Returns
        -------
        dict
            A dictionary where the keys are the metric names and the values are the calculated metric values.
        """
        from pyspark.ml.evaluation import RegressionEvaluation

        results = {}
        for metric in metrics:
            evaluator = RegressionEvaluation(labelCol=labelCol, predictionCol=predictionCol, metricName=metric, **kwargs)
            value = evaluator.evaluate(predictions)
            results[metric] = value
        return results
