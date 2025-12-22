import numpy as np
from scipy.interpolate import interp1d
import geopandas as gpd
import matplotlib.pyplot as plt

def wanda_to_shape(
        points: np.ndarray,
        npoints: int,
        kind: str = "linear"
):
    """
    Calculate the intermediate points with an arc-length method. The interpolation solves the following equation
    f(s) = (x,y) with s the distance along the curve.

    Parameters
    ----------
    points: np.array([points.shape[0], 2])
        The coordinates of the linestring.
    npoints: int
        The number of points to interpolate to (i.e., wanda computational nodes).
    kind: str (linear)
        Type of interpolation method.

    Returns
    -------
    interpolated_points: np.ndarray
        An array with the interpolated points as np.array([npoints, 2])

    Source
    ------
    [1] https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
    [2] https://stackoverflow.com/questions/35673895/type-hinting-annotation-pep-484-for-numpy-ndarray

    """
    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    slength = distance[-1]
    distance = np.insert(distance, 0, 0) / slength
    # Create interpolator object
    interpolator = interp1d(distance, points, kind=kind, axis=0)
    # Interpolate points
    sdistance = np.linspace(0, 1, npoints)
    interpolated_points = interpolator(sdistance)
    return np.c_[interpolated_points, sdistance * slength]


def plot_interpolation_wanda_to_schape(
        points: np.ndarray,
        interpolated_points: np.ndarray,
        export: str = None
):
    """
    Evaluate the performance of the interpolation.

    Parameters
    ----------
    points: np.ndarray
        Original coordinates of the shape-file
    interpolated_points: np.ndarray
        Interpolated coordinates of the shape-file
    export: str (None)
        Save file for reference to the path specified.

    Returns
    -------

    """
    plt.figure(figsize=(15/2.54, 12/2.54))
    plt.plot(interpolated_points[:, 0], interpolated_points[:, 1], '-k',
             label='Interpolated points')
    plt.plot(points[:, 0], points[:, 1], 'or', markersize=3,
             label='Original points')

    plt.legend(frameon=False)
    plt.xlabel("x-distance (-)")
    plt.ylabel("y-distance (-)")
    plt.tight_layout()
    if export is not None:
        plt.savefig(export)