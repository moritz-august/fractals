from enum import Enum

import torch


class VisualizationMode(Enum):
    CLICK = 'click'
    AUTO = 'auto'


def normalize_min_max(frame: torch.Tensor) -> torch.Tensor:
    frame -= frame.min()
    frame /= frame.max()

    return frame


def get_complex_plane(center: complex, radius: float, resolution: int) -> torch.Tensor:
    reals = torch.linspace(center.real - radius, center.real + radius, resolution)
    imags = torch.linspace(center.imag - radius, center.imag + radius, resolution)
    c_plane = torch.zeros((resolution, resolution), dtype=torch.complex128)
    c_plane.real = torch.tile(reals.unsqueeze(0), (resolution, 1))
    c_plane.imag = torch.tile(imags.unsqueeze(0), (resolution, 1)).transpose(0, 1)

    return c_plane
