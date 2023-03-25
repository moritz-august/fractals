import numpy as np
import torch

from util import get_complex_plane
from util import normalize_const

HAS_CUDA = torch.cuda.is_available()


def mandelbrot_fn(c: torch.Tensor, depth: int = 100, bailout: float = 2 ** 8) -> torch.Tensor:
    c = torch.as_tensor(c, device=torch.device('cuda') if HAS_CUDA else torch.device('cpu'))
    z = torch.zeros_like(c)
    counts = torch.zeros_like(c, dtype=torch.float)
    continuous_counts = torch.zeros_like(counts, dtype=torch.double)
    update_mask = torch.abs(z) <= bailout

    for _ in range(depth):
        z[update_mask] = (z * z + c)[update_mask]
        update_mask = torch.abs(z) <= bailout
        counts[update_mask] += 1

    depth_mask = counts < depth
    continuous_counts[depth_mask] = z[depth_mask].abs().log() / 2
    continuous_counts[depth_mask] = (continuous_counts[depth_mask] / np.log(2)).log() / np.log(2)
    continuous_counts[depth_mask] = counts[depth_mask] + 1 - continuous_counts[depth_mask]

    return continuous_counts


def compute_frame(center: complex = complex(0., 0.), radius: float = 2., resolution: int = 200,
                  depth: int = 100) -> np.ndarray:
    c_plane = get_complex_plane(center, radius, resolution)
    frame = mandelbrot_fn(c_plane)
    frame = normalize_const(frame, depth)

    return frame.cpu().numpy()
