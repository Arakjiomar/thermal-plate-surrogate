"""Neural surrogate for steady 2D heat conduction on a plate."""
from .solver import PlateParams, solve_plate, solve_poisson, make_grid, laplacian_matrix

__all__ = ["PlateParams", "solve_plate", "solve_poisson", "make_grid", "laplacian_matrix"]
__version__ = "0.1.0"
