from enum import Enum

import torch


class FractalClass(Enum):
    MANDELBROT = 'mandelbrot'
    JULIA = 'julia'


class VisualizationMode(Enum):
    CLICK = 'click'
    AUTO = 'auto'


def normalize_min_max(frame: torch.Tensor) -> torch.Tensor:
    frame -= frame.min()
    frame /= frame.max()

    return frame


def normalize_const(frame: torch.Tensor, const: float) -> torch.Tensor:
    return frame / const


def get_complex_plane(center: complex, radius: float, resolution: int) -> torch.Tensor:
    reals = torch.linspace(center.real - radius, center.real + radius, resolution, dtype=torch.float64)
    imags = torch.linspace(center.imag - radius, center.imag + radius, resolution, dtype=torch.float64)
    c_plane = torch.zeros((resolution, resolution), dtype=torch.complex128)
    c_plane.real = torch.tile(reals.unsqueeze(0), (resolution, 1))
    c_plane.imag = torch.tile(imags.unsqueeze(0), (resolution, 1)).transpose(0, 1)

    return c_plane
