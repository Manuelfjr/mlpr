"""
Module for uncertainty plots on supervisioned learning.
"""

from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class UncertaintyPlots:
    """
    Class for plotting uncertainty related to classification models.

    Methods
    -------
    uncertainty(probs, X, figsize, grid_layout=(4, 2), show_inline=True, box_on=True, **kwargs)
        Plots a scatter plot for the given probabilities.

    uncertainty_x(probs, X, figsize, num_cols=None, show_inline=True, box_on=True, **kwargs)
        Plots a scatter plot for the given probabilities.
    """

    def __init__(self, default_figsize: Tuple[int, int] = (10, 10)):
        self.default_figsize: Tuple[int, int] = default_figsize

    def single_model_uncertainty(
        self,
        model_name: str,
        prob: Union[pd.DataFrame, pd.Series, np.ndarray],
        X: Union[pd.DataFrame, np.ndarray],
        figsize: Tuple[int, int],
        show_inline: bool = True,
        box_on: bool = True,
        **kwargs,
    ) -> Union[Tuple[plt.Figure, np.ndarray], None]:
        """
        Plots a scatter plot for the given probabilities for a single model.

        Parameters
        ----------
        model_name : str
            The name of the model to plot.
        prob : DataFrame
            Probabilities for the model.
        X : DataFrame or ndarray
            Test data to be plotted.
        figsize : tuple of (int, int)
            Size of the figure to be plotted.
        show_inline : bool, default=True
            Whether the plot should be shown inline.
        box_on : bool, default=True
            Whether the axes of the plot should be shown.
        **kwargs : dict
            Additional keyword arguments to be passed to the scatter plot.

        Returns
        -------
        tuple of (Figure, ndarray) or None
            A tuple containing the figure and the axes of the plot, or None if `show_inline` is False.
        """
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=prob, **kwargs)
        ax.set_title(f"{model_name}")
        fig.colorbar(scatter, ax=ax)
        ax.set_frame_on(box_on)
        if not box_on:
            ax.set_xticks([])
            ax.set_yticks([])
        if show_inline:
            plt.show()
        else:
            plt.close()
        return fig, ax

    def uncertainty(
        self,
        model_names: List[List[str]],
        probs: Dict[str, pd.DataFrame],
        X: Union[pd.DataFrame, np.ndarray],
        figsize: Tuple[int, int],
        show_inline: bool = True,
        box_on: bool = True,
        **kwargs,
    ) -> Union[Tuple[plt.Figure, np.ndarray], None]:
        """
        Plots a grid of scatter plots for the given probabilities.

        Parameters
        ----------
        model_names : list of list of str
            The names of the models to plot. Each sublist corresponds to a row in the grid.
        probs : dict of {str: DataFrame}
            Dictionary of probabilities with names as keys and DataFrames as values.
        X : DataFrame or ndarray
            Test data to be plotted.
        figsize : tuple of (int, int)
            Size of the figure to be plotted.
        show_inline : bool, default=True
            Whether the plot should be shown inline.
        box_on : bool, default=True
            Whether the axes of the plot should be shown.
        **kwargs : dict
            Additional keyword arguments to be passed to the scatter plot.

        Returns
        -------
        tuple of (Figure, ndarray) or None
            A tuple containing the figure and the axes of the plot, or None if `show_inline` is False.
        """
        grid_size: tuple[int, int] = len(model_names), max(map(len, model_names))

        fig_inc, ax_inc = plt.subplots(*grid_size, figsize=figsize)

        if grid_size[0] == 1 and grid_size[1] == 1:
            ax_inc: np.ndarray[Any, np.dtype[Any]] = np.array([[ax_inc]])
        elif grid_size[0] == 1 or grid_size[1] == 1:
            ax_inc = ax_inc.reshape(grid_size)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if j < len(model_names[i]):
                    prob: pd.DataFrame = probs.get(model_names[i][j])
                    if prob is not None:
                        scatter = ax_inc[i, j].scatter(X[:, 0], X[:, 1], c=prob, **kwargs)
                        ax_inc[i, j].set_title(f"{model_names[i][j]}")
                        fig_inc.colorbar(scatter, ax=ax_inc[i, j])
                        ax_inc[i, j].set_frame_on(box_on)
                        if not box_on:
                            ax_inc[i, j].set_xticks([])
                            ax_inc[i, j].set_yticks([])
                    else:
                        ax_inc[i, j].set_visible(False)
                else:
                    ax_inc[i, j].set_visible(False)

        fig_inc.tight_layout()
        if show_inline:
            plt.show()
        else:
            plt.close()
        return fig_inc, ax_inc
