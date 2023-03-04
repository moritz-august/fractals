from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


def mandelbrot_fn(c: np.ndarray, n_iterations: int = 100) -> np.ndarray:
    z = np.zeros_like(c)
    counts = np.zeros_like(c, dtype=float)
    for _ in range(n_iterations):
        z = z * z + c
        counts[np.abs(z) <= 2] += 1
    return counts


def compute_frame(center: complex = complex(0, 0), radius: float = 2., resolution: int = 200) -> np.ndarray:
    reals = np.linspace(center.real - radius, center.real + radius, resolution)
    imags = np.linspace(center.imag - radius, center.imag + radius, resolution)
    c_plane = np.zeros((resolution, resolution), dtype=complex)
    c_plane.real = np.tile(reals[np.newaxis, ...], (resolution, 1))
    c_plane.imag = np.tile(imags[np.newaxis, ...], (resolution, 1)).transpose()

    frame = mandelbrot_fn(c_plane)

    frame -= frame.min()
    frame /= frame.max()

    return frame


def update_domain_params(center_real: float, center_imag: float, real_coord: float, imag_coord: float,
                         zoom_fac: float = 1.2) -> Tuple[float, float]:
    global radius, res
    min_real = center_real - radius
    min_imag = center_imag - radius
    center_real = min_real + real_coord / res * 2 * radius
    center_imag = min_imag + imag_coord / res * 2 * radius
    radius /= zoom_fac

    return center_real, center_imag


def visualize_mandelbrot_interactive(resolution: int = 200, start_radius: float = 2., start_real: float = 0,
                                     start_imag: float = 0):
    global ax, res
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot()

    global radius, center_real, center_imag
    res = resolution
    radius = start_radius
    center_real = start_real
    center_imag = start_imag

    def onclick(event):
        global center_real, center_imag
        center_real, center_imag = update_domain_params(center_real, center_imag, event.xdata, event.ydata)
        frame = compute_frame(complex(center_real, center_imag), radius, res)
        ax.clear()
        ax.imshow(frame, cmap='viridis')
        plt.gcf().canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)

    frame = compute_frame(complex(center_real, center_imag), radius)
    ax.imshow(frame, cmap='viridis')
    ax.set_xticks([])
    ax.set_yticks([])

    while True:
        fig.canvas.draw()
        fig.canvas.flush_events()


def visualize_mandelbrot_automatic(resolution: int = 200, start_radius: float = 2., start_real: float = 0,
                                   start_imag: float = 0, zoom_fac: float = 1.05, center_update_freq: int = 20):
    global ax, res
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot()

    global radius
    res = resolution
    radius = start_radius
    center_real = start_real
    center_imag = start_imag

    frame = compute_frame(complex(center_real, center_imag), radius)
    ax.imshow(frame, cmap='viridis')
    ax.set_xticks([])
    ax.set_yticks([])

    center_imag_coords, center_real_coords = np.where(np.abs(frame - 0.5) < 0.1)
    sampled_coord_idx = np.random.choice(np.arange(len(center_real_coords)))
    center_real_coord = center_real_coords[sampled_coord_idx]
    center_imag_coord = center_imag_coords[sampled_coord_idx]

    i = 0
    while True:
        center_real, center_imag = update_domain_params(center_real, center_imag, center_real_coord, center_imag_coord, zoom_fac)
        frame = compute_frame(complex(center_real, center_imag), radius, res)
        ax.clear()
        ax.imshow(frame, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.draw()
        fig.canvas.flush_events()

        if i % center_update_freq == 0:
            center_imag_coords, center_real_coords = np.where(np.abs(frame - 0.5) < 0.1)
            if len(center_real_coords) == 0:
                break
            sampled_coord_idx = np.argmin([np.sqrt(
                (center_real - center_real_coord) ** 2 + (center_imag - center_imag_coord) ** 2)
                for center_real, center_imag in
                zip(center_real_coords, center_imag_coords)])
            center_real_coord = center_real_coords[sampled_coord_idx]
            center_imag_coord = center_imag_coords[sampled_coord_idx]
        else:
            center_real_coord = res / 2
            center_imag_coord = res / 2

        i += 1


if __name__ == '__main__':
    # visualize_mandelbrot_interactive()
    visualize_mandelbrot_automatic()
