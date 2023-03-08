from enum import Enum

import numpy as np


class VisualizationMode(Enum):
    CLICK = 'click'
    AUTO = 'auto'


def normalize_min_max(frame: np.ndarray) -> np.ndarray:
    frame -= frame.min()
    frame /= frame.max()

    return frame


def get_complex_plane(center: complex, radius: float, resolution: int) -> np.ndarray:
    reals = np.linspace(center.real - radius, center.real + radius, resolution)
    imags = np.linspace(center.imag - radius, center.imag + radius, resolution)
    c_plane = np.zeros((resolution, resolution), dtype=complex)
    c_plane.real = np.tile(reals[np.newaxis, ...], (resolution, 1))
    c_plane.imag = np.tile(imags[np.newaxis, ...], (resolution, 1)).transpose()

    return c_plane
