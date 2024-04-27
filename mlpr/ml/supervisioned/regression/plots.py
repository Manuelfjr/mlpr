from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib as mpl


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

    def __init__(self, data: pd.DataFrame, color_palette: Optional[List[str]] = None):
        self.data = data
        self.color_palette = color_palette
        self.original_prop_cycle = None
        if self.color_palette is not None:
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=self.color_palette)


    def check_if_inline(self, show_inline: bool):
        if not show_inline:
            plt.close()
            return None

    def _is_grid(self, ax: Optional[Axes], figsize: Tuple[int, int]):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        return fig, ax

    def check_color_map(self, step: str = "before"):
        if (self.color_palette is not None) and (step == "before"):
            self.original_prop_cycle = mpl.rcParams['axes.prop_cycle']
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=self.color_palette)
        elif (self.color_palette is not None) and (step == "after"):
            mpl.rcParams['axes.prop_cycle'] = self.original_prop_cycle
        return None
    
    def scatter(
        self,
        y_true_col: str,
        y_pred_col: str,
        label: Optional[str]=None,
        ax: Optional[Axes]=None,
        figsize: Tuple[int, int]=(12, 6),
        linestyle: str="--",
        linecolor: str="r",
        show_inline: bool = False,
        corr_pos_x: float = 0.05,
        corr_pos_y: float = 0.95,
        **kwargs: Dict[str, Any]
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
        **kwargs : dict
            Additional keyword arguments to pass to `ax.scatter`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the scatter plot.
        ax : matplotlib.axes.Axes
            The axes containing the scatter plot.
        """
        p1 = max(max(self.data[y_true_col]), max(self.data[y_pred_col]))
        p2 = min(min(self.data[y_true_col]), min(self.data[y_pred_col]))

        self.check_color_map()
        fig, ax = self._is_grid(ax, figsize=figsize)

        ax.scatter(self.data[y_true_col], self.data[y_pred_col], label=label, **kwargs)
        ax.plot([p1, p2], [p1, p2], linestyle, color=linecolor)

        # Calculate the correlation
        corr = self.data[y_true_col].corr(self.data[y_pred_col])

        # Add the correlation to the plot
        ax.text(corr_pos_x, corr_pos_y, f'Correlation: {corr:.2f}', transform=ax.transAxes, verticalalignment='top')

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
        **kwargs: Dict[str, Any]
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

        ks_stats, pvalue = ks_2samp(self.data[y_true_col], self.data[y_pred_col])

        self.check_color_map()
        fig, ax = self._is_grid(ax, figsize=figsize)

        ax.plot(ecdf_true.x, ecdf_true.y, label=y_true_col, **kwargs)
        ax.plot(ecdf_pred.x, ecdf_pred.y, label=y_pred_col, **kwargs)

        ax.set_ylabel('ECDF')
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
        **kwargs: Dict[str, Any]
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
        data = self.data[columns]
        kde = {}

        self.check_color_map()
        fig, ax = self._is_grid(ax, figsize=figsize)

        for col in columns:
            kde[col] = gaussian_kde(data[col])
            x = np.linspace(min(data[col]), max(data[col]), 1000)
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
        linestyle: str="--",
        linecolor: str="r",
        show_inline: bool = False,
        **kwargs: Dict[str, Any]
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

        self.error = self.data[y_true_col] - self.data[y_pred_col]
        self.std_error = (self.error - np.mean(self.error)) / np.std(self.error)

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
            plot_functions: List[List[str]] = [['scatter', 'plot_ecdf'], ['plot_kde', 'plot_error_hist']],
            plot_args: Dict[str, Dict[str, Any]] = {},
            figsize: Tuple[int, int] = (18, 12),
            **kwargs: Dict[str, Any]
        ) -> Tuple[Figure, Axes]:
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
        max_cols = max(map(len, plot_functions))
        grid_size = len(plot_functions), max_cols

        fig, axs = plt.subplots(*grid_size, figsize=figsize)
        
        self.check_color_map()

        # Ensure axs is always a 2D array
        if grid_size[0] == 1 and grid_size[1] == 1:
            axs = np.array([[axs]])
        elif grid_size[0] == 1 or grid_size[1] == 1:
            axs = axs.reshape(grid_size)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if j < len(plot_functions[i]):
                    func_name = plot_functions[i][j]
                    func = getattr(self, func_name)
                    args = {**kwargs, **plot_args.get(func_name, {})}
                    func(ax=axs[i, j], **args)
                else:
                    # If there is no corresponding plot function, hide the axis
                    axs[i, j].axis('off')

        self.check_color_map("after")

        fig.tight_layout()

        return fig, axs