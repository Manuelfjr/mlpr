from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde, ks_2samp
from statsmodels.distributions.empirical_distribution import ECDF
from cycler import cycler
from matplotlib import rcParams
# 

class RegressionPlots:
    """
    A class for creating various plots for regression analysis.

    Attributes
    ----------
    data : pd.DataFrame
        The data to be plotted.
    color_palette: list, optional
        The color palette to use for the plots. If None, the default color palette is used.
    """
    def __init__(self, data: pd.DataFrame, color_palette: Optional[List[str]] = None) -> None:
        self.data: pd.DataFrame = data
        self.color_palette: List[str] | None = color_palette
        self.original_prop_cycle = None
        if self.color_palette is not None:
            rcParams["axes.prop_cycle"] = cycler(color=self.color_palette)

    def __search(
        self, metrics_dict: dict = {}, intervals_dict: dict = {}, metric: str = "precision", positive: bool = True
    ) -> dict:
        method: Dict[Any, Any]= {
            i_class: content["metrics"][metric][0] if len(
                content["metrics"][metric]
                ) == 1 else content["metrics"][metric][int(positive)] 
            for i_class, content in metrics_dict.items()
        }
        k_worst_class: Union[int, str] = min(method, key=method.get)  # type: ignore
        return intervals_dict[k_worst_class]

    def __set_vxlines(
            self,
            ax: Axes,
            interval: Tuple[float, float],
            xy: Optional[int]=None,
            color: str='black',
            linestyle: str='--'
        ) -> Axes:
        if np.isneginf(interval[0]):
            ax.axvline(x=interval[1], color=color, linestyle=linestyle)
        elif np.isposinf(interval[1]):
            ax.axvline(x=interval[0], color=color, linestyle=linestyle)
        else:
            ax.axvline(x=interval[0], color=color, linestyle=linestyle)
            ax.axvline(x=interval[1], color=color, linestyle=linestyle)
        if xy is not None:
            ax.axvline(x=xy, color=color, linestyle=linestyle)
        return ax

    def check_if_inline(self, show_inline: bool) -> None:
        if not show_inline:
            plt.close()
            return None

    def _is_grid(self, ax: Optional[Axes], figsize: Tuple[int, int]) -> Tuple[Union[Figure, None], Union[Axes, None]]:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig: Figure | None = ax.get_figure()
        return fig, ax

    def check_color_map(self, step: str = "before") -> None:
        if (self.color_palette is not None) and (step == "before"):
            self.original_prop_cycle: Any = rcParams["axes.prop_cycle"]
            rcParams["axes.prop_cycle"] = cycler(color=self.color_palette)
        elif (self.color_palette is not None) and (step == "after"):
            rcParams["axes.prop_cycle"] = self.original_prop_cycle
        return None
    
    def plot_fitted(
        self,
        y_true_col: str,
        y_pred_col: str,
        condition: pd.Series,
        seed: int=42,
        sample_size: Optional[int] = None,
        ax: Optional[Axes] = None,
        figsize: Tuple[int, int] = (12, 6),
        show_inline: bool = False,
        **kwargs
    ) -> Axes:
        """
        Plot the true and predicted values from the data DataFrame.

        Parameters
        ----------
        y_true_col : str
            The column name of the true values in the data DataFrame.
        y_pred_col : str
            The column name of the predicted values in the data DataFrame.
        condition : pd.Series
            The condition to filter the data.
        sample_size : int, optional
            The number of samples to draw from the data. If None, all data is used. Default is None.
        ax : matplotlib.axes.Axes, optional
            The axes upon which to plot. If None, a new figure and axes are created. Default is None.
        figsize : Tuple[int, int], optional
            The size of the figure (width, height) in inches. Default is (12, 6).
        show_inline : bool, optional
            Whether to display the plot inline in Jupyter Notebook or IPython. Default is False.
        **kwargs : dict
            Additional keyword arguments to pass to plt.plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes on which the data was plotted.
        """
        self.check_color_map()
        fig, ax = self._is_grid(ax, figsize=figsize)

        if condition is not None:
            filtered_data: pd.DataFrame = self.data[condition].copy()
        else:
            filtered_data = self.data.copy()

        if sample_size is not None:
            filtered_data = filtered_data.sample(sample_size, random_state=seed)

        ax.plot(list(filtered_data[y_true_col].values), label=y_true_col, **kwargs)
        ax.plot(list(filtered_data[y_pred_col].values), label=y_pred_col, **kwargs)
        ax.legend()
        ax.grid(True)

        self.check_if_inline(show_inline)
        self.check_color_map("after")

        fig.tight_layout()

        return ax

    def scatter(
        self,
        y_true_col: str,
        y_pred_col: str,
        label: Optional[str] = None,
        ax: Optional[Axes] = None,
        figsize: Tuple[int, int] = (12, 6),
        linestyle: str = "--",
        linecolor: str = "r",
        show_inline: bool = False,
        corr_pos_x: float = 0.05,
        corr_pos_y: float = 0.95,
        worst_interval: bool = False,
        linestyle_interval: str = "--",
        linecolor_interval: str = "black",
        metrics: dict = {},
        class_interval: dict = {},
        method: str = "precision",
        positive: bool = True,
        **kwargs: Dict[str, Any],
    ) -> Tuple[Figure, Axes]:
        """
        Create a scatter plot of the true vs predicted values.

        Parameters
        ----------
        y_true_col : str
            The name of the column containing the true values.
        y_pred_col : str
            The name of the column containing the predicted values.
        label : str, optional
            The label for the data points in the scatter plot.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        figsize : tuple, optional
            The size of the figure to create. Ignored if `ax` is not None.
        linestyle : str, optional
            The line style for the diagonal line in the scatter plot.
        linecolor : str, optional
            The line color for the diagonal line in the scatter plot.
        show_inline : bool, optional
            If True, the plot is displayed inline. Default is False.
        corr_pos_x : float, optional
            The x position of the correlation text in the plot. Default is 0.05.
        corr_pos_y : float, optional
            The y position of the correlation text in the plot. Default is 0.95.
        worst_interval : bool, optional
            If True, plot the worst interval. Default is False.
        linestyle_interval : str, optional
            The line style for the worst interval. Default is "--".
        linecolor_interval : str, optional
            The line color for the worst interval. Default is "black".
        metrics : dict, optional
            A dictionary of metrics.
        class_interval : dict, optional
            A dictionary of class intervals.
        method : str, optional
            The method used to calculate the worst interval. Default is "precision".
        positive : bool, optional
            If True, calculate the worst interval for positive class. Default is True.
        **kwargs : dict
            Additional keyword arguments to pass to `ax.scatter`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the scatter plot.
        ax : matplotlib.axes.Axes
            The axes containing the scatter plot.
        """

        p1: float | int = max(max(self.data[y_true_col]), max(self.data[y_pred_col]))
        p2: float | int = min(min(self.data[y_true_col]), min(self.data[y_pred_col]))

        self.check_color_map()
        fig, ax = self._is_grid(ax, figsize=figsize)

        ax.scatter(self.data[y_true_col], self.data[y_pred_col], label=label, **kwargs)
        ax.plot([p1, p2], [p1, p2], linestyle, color=linecolor)

        if worst_interval:
            self._worst_interval: tuple = self.__search(metrics, class_interval, method, positive)
            xy: float | int | None = p2 if -np.inf in self._worst_interval else p1 if np.inf in self._worst_interval else None
            ax = self.__set_vxlines(ax, self._worst_interval, xy, linecolor_interval, linestyle_interval)

        corr: float = self.data[y_true_col].corr(self.data[y_pred_col])

        ax.text(corr_pos_x, corr_pos_y, f"Correlation: {corr:.2f}", transform=ax.transAxes, verticalalignment="top")

        ax.set_xlabel(y_true_col)
        ax.set_ylabel(y_pred_col)

        if label:
            ax.legend()

        ax.grid(True)

        self.check_if_inline(show_inline)
        self.check_color_map("after")

        fig.tight_layout()

        return fig, ax

    def plot_ecdf(
        self,
        y_true_col: str,
        y_pred_col: str,
        ax: Optional[Axes] = None,
        figsize: Tuple[int, int] = (12, 6),
        show_inline: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Tuple[Figure, Axes]:
        """
        Plot the empirical cumulative distribution function (ECDF) of the true and predicted values.

        Parameters
        ----------
        y_true_col : str
            The name of the column containing the true values.
        y_pred_col : str
            The name of the column containing the predicted values.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        figsize : tuple, optional
            The size of the figure to create. Ignored if `ax` is not None.
        show_inline : bool, optional
            If True, the plot is displayed inline. Default is False.
        **kwargs : dict
            Additional keyword arguments to pass to `ax.plot`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot.
        ax : matplotlib.axes.Axes
            The axes containing the plot.
        """
        ecdf_true = ECDF(self.data[y_true_col])
        ecdf_pred = ECDF(self.data[y_pred_col])

        ks_stats, _ = ks_2samp(self.data[y_true_col], self.data[y_pred_col])

        self.check_color_map()
        fig, ax = self._is_grid(ax, figsize=figsize)

        ax.plot(ecdf_true.x, ecdf_true.y, label=y_true_col, **kwargs)
        ax.plot(ecdf_pred.x, ecdf_pred.y, label=y_pred_col, **kwargs)

        ax.set_ylabel("ECDF")
        ax.set_title(f"ks_stats: {ks_stats:.2f}")

        ax.legend()
        ax.grid(True)

        self.check_if_inline(show_inline)
        self.check_color_map("after")

        fig.tight_layout()

        return fig, ax

    def plot_kde(
        self,
        columns: List[str],
        ax: Optional[Axes] = None,
        figsize: Tuple[int, int] = (12, 6),
        show_inline: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Tuple[Figure, Axes]:
        """
        Plot the kernel density estimate (KDE) for the specified columns.

        Parameters
        ----------
        columns : list of str
            The names of the columns to plot.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        figsize : tuple, optional
            The size of the figure to create. Ignored if `ax` is not None.
        show_inline : bool, optional
            If True, the plot is displayed inline. Default is False.
        **kwargs : dict
            Additional keyword arguments to pass to `ax.plot`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot.
        ax : matplotlib.axes.Axes
            The axes containing the plot.
        """
        data: pd.DataFrame = self.data[columns]
        kde: dict = {}

        self.check_color_map()
        fig, ax = self._is_grid(ax, figsize=figsize)

        for col in columns:
            kde[col] = gaussian_kde(data[col])
            x: Any = np.linspace(min(data[col]), max(data[col]), 1000)
            ax.plot(x, kde[col](x), label=col, **kwargs)

            ax.set_ylabel(col)
            ax.legend()
            ax.grid(True)

        self.check_if_inline(show_inline)
        self.check_color_map("after")

        fig.tight_layout()

        return fig, ax

    def plot_error_hist(
        self,
        y_true_col: str,
        y_pred_col: str,
        label: Optional[str] = None,
        ax: Optional[Axes] = None,
        figsize: Tuple[int, int] = (12, 6),
        linestyle: str = "--",
        linecolor: str = "r",
        show_inline: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Tuple[Figure, Axes]:
        """
        Plot a histogram of the error between the true and predicted values.

        Parameters
        ----------
        y_true_col : str
            The name of the column containing the true values.
        y_pred_col : str
            The name of the column containing the predicted values.
        label : str, optional
            The label for the histogram.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.
        figsize : tuple, optional
            The size of the figure to create. Ignored if `ax` is not None.
        linestyle : str, optional
            The line style for the vertical line indicating the mean error. Default is "--".
        linecolor : str, optional
            The line color for the vertical line indicating the mean error. Default is "r".
        show_inline : bool, optional
            If True, the plot is displayed inline. Default is False.
        **kwargs : dict
            Additional keyword arguments to pass to `ax.hist`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot.
        ax : matplotlib.axes.Axes
            The axes containing the plot.
        """
        fig, ax = self._is_grid(ax, figsize=figsize)

        self.error: pd.Series = self.data[y_true_col] - self.data[y_pred_col]
        self.std_error: Union[np.ndarray, pd.Series] = (self.error - np.mean(self.error)) / np.std(self.error)

        self.check_color_map()
        ax.hist(self.std_error, label=label, **kwargs)

        ax.axvline(x=0, color=linecolor, linestyle=linestyle)
        ax.grid(True)

        self.check_if_inline(show_inline)
        self.check_color_map("after")

        fig.tight_layout()

        return fig, ax

    def grid_plot(
        self,
        plot_functions: List[List[str]],
        plot_args: Dict[str, Dict[str, Any]] = {},
        figsize: tuple[int, int] = (18, 12),
        **kwargs: Dict[str, Any],
    ) -> tuple[Figure, Union[Axes, np.ndarray]]:
        """
        Plot a grid of plots using the specified plot functions.

        Parameters
        ----------
        plot_functions : list of list of str
            The names of the plot functions to use. Each sublist corresponds to a row in the grid.
        plot_args : dict, optional
            Additional arguments to pass to the plot functions. The keys should be the names of the plot functions.
        figsize : tuple, optional
            The size of the figure to create.
        **kwargs : dict
            Additional keyword arguments to pass to the plot functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot.
        axs : array of matplotlib.axes.Axes
            The axes containing the plots.
        """
        max_cols: int = max(map(len, plot_functions))
        grid_size: tuple[int, int] = len(plot_functions), max_cols

        fig, axs = plt.subplots(*grid_size, figsize=figsize) 

        self.check_color_map()

        if grid_size[0] == 1 and grid_size[1] == 1:
            axs: np.ndarray[Any, np.dtype[Any]] = np.array([[axs]])
        elif grid_size[0] == 1 or grid_size[1] == 1:
            axs = axs.reshape(grid_size)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if j < len(plot_functions[i]):
                    graph_name: str = plot_functions[i][j]
                    graph_info: Dict[str, Any] | None = plot_args.get(graph_name)
                    if graph_info:
                        func_name: Any = graph_info["plot"]
                        func: Any = getattr(self, func_name)
                        args: Dict[str, Dict[str, Any]] = {**kwargs, **graph_info["params"]}
                        func(ax=axs[i, j], **args)
                    else:
                        axs[i, j].axis("off")
                else:
                    axs[i, j].set_visible(False)

        self.check_color_map("after")

        fig.tight_layout()

        return fig, axs