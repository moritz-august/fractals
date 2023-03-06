import numpy as np

from util import normalize_min_max


def mandelbrot_fn(c: np.ndarray, depth: int = 100) -> np.ndarray:
    z = np.zeros_like(c)
    counts = np.zeros_like(c, dtype=float)
    update_mask = np.abs(z) <= 2.

    for _ in range(depth):
        z[update_mask] = (z * z + c)[update_mask]
        update_mask = np.abs(z) <= 2.
        counts[update_mask] += 1

    return counts


def compute_frame(center: complex = complex(0., 0.), radius: float = 2., resolution: int = 200) -> np.ndarray:
    reals = np.linspace(center.real - radius, center.real + radius, resolution)
    imags = np.linspace(center.imag - radius, center.imag + radius, resolution)
    c_plane = np.zeros((resolution, resolution), dtype=complex)
    c_plane.real = np.tile(reals[np.newaxis, ...], (resolution, 1))
    c_plane.imag = np.tile(imags[np.newaxis, ...], (resolution, 1)).transpose()

    frame = mandelbrot_fn(c_plane)
    frame = normalize_min_max(frame)

    return frame
