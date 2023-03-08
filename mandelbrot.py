import numpy as np
import torch

from util import get_complex_plane
from util import normalize_min_max

HAS_CUDA = torch.cuda.is_available()


def mandelbrot_fn(c: torch.Tensor, depth: int = 100) -> torch.Tensor:
    c = torch.as_tensor(c, device=torch.device('cuda') if HAS_CUDA else torch.device('cpu'))
    z = torch.zeros_like(c)
    counts = torch.zeros_like(c, dtype=torch.float)
    update_mask = torch.abs(z) <= 2.

    for _ in range(depth):
        z[update_mask] = (z * z + c)[update_mask]
        update_mask = torch.abs(z) <= 2.
        counts[update_mask] += 1

    return counts


def compute_frame(center: complex = complex(0., 0.), radius: float = 2., resolution: int = 200) -> np.ndarray:
    c_plane = get_complex_plane(center, radius, resolution)
    frame = mandelbrot_fn(c_plane)
    frame = normalize_min_max(frame)

    return frame.cpu().numpy()
