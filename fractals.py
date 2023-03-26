from typing import Callable

import numpy as np
import torch

from util import get_complex_plane
from util import normalize_const

HAS_CUDA = torch.cuda.is_available()


def mandelbrot(c: torch.Tensor, depth: int = 100, bailout: float = 2 ** 8) -> torch.Tensor:
    c = torch.as_tensor(c, device=torch.device('cuda') if HAS_CUDA else torch.device('cpu'))
    z = torch.zeros_like(c)

    counts = get_counts(c, z, depth, bailout)
    continuous_counts = get_continuous_counts(z, counts, depth)

    return continuous_counts


def julia(z: torch.Tensor, c: complex = -0.8 + 0.156j, depth: int = 100, bailout: float = 2 ** 8) -> torch.Tensor:
    z = torch.as_tensor(z, device=torch.device('cuda') if HAS_CUDA else torch.device('cpu'))
    c = torch.ones_like(z) * c

    counts = get_counts(c, z, depth, bailout)
    continuous_counts = get_continuous_counts(z, counts, depth)

    return continuous_counts


def get_counts(c: torch.Tensor, z: torch.Tensor, depth: int, bailout: float) -> torch.Tensor:
    counts = torch.zeros_like(c, dtype=torch.float)
    update_mask = torch.abs(z) <= bailout
    for _ in range(depth):
        z[update_mask] = (z * z + c)[update_mask]
        update_mask = torch.abs(z) <= bailout
        counts[update_mask] += 1
    return counts


def get_continuous_counts(z: torch.Tensor, counts: torch.Tensor, depth: int) -> torch.Tensor:
    depth_mask = counts < depth
    continuous_counts = torch.ones_like(counts, dtype=torch.double) * depth
    continuous_counts[depth_mask] = z[depth_mask].abs().log() / 2
    continuous_counts[depth_mask] = (continuous_counts[depth_mask] / np.log(2)).log() / np.log(2)
    continuous_counts[depth_mask] = counts[depth_mask] + 1 - continuous_counts[depth_mask]
    return continuous_counts


def compute_frame(center: complex = complex(0., 0.), radius: float = 2., resolution: int = 200,
                  fractal_fn: Callable = julia, depth: int = 100) -> np.ndarray:
    c_plane = get_complex_plane(center, radius, resolution)
    frame = fractal_fn(c_plane)
    frame = normalize_const(frame, depth)

    return frame.cpu().numpy()
