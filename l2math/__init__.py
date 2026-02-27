"""L2Math - Utility functions for L2 Mathematics practicals."""

from l2math.visualization import (
    plot_gradient_descent_1d,
    plot_loss_history,
    plot_predictions,
    plot_gradient_step,
    plot_coefficient_path,
)
from l2math.display import print_model_params
from l2math.data import generate_linear_data

__all__ = [
    "plot_gradient_descent_1d",
    "plot_loss_history",
    "plot_predictions",
    "plot_gradient_step",
    "plot_coefficient_path",
    "print_model_params",
    "generate_linear_data",
]
