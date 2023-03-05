from argparse import ArgumentParser
from enum import Enum
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from mandelbrot import compute_frame


def get_center_candidate_coords(frame: np.ndarray, target_val: float = 0.5, tolerance: float = 0.1) -> np.ndarray:
    return np.where(np.abs(frame - target_val) < tolerance)


def update_domain_params(center_real: float, center_imag: float, real_coord: float, imag_coord: float,
                         zoom_fac: float = 1.2) -> Tuple[float, float]:
    global radius, res
    min_real = center_real - radius
    min_imag = center_imag - radius
    center_real = min_real + real_coord / res * 2 * radius
    center_imag = min_imag + imag_coord / res * 2 * radius
    radius /= zoom_fac

    return center_real, center_imag


def zoom_mandelbrot_click(resolution: int = 200, zoom_fac: float = 1.2, start_radius: float = 2., start_real: float = 0,
                          start_imag: float = 0):
    global res
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
        center_real, center_imag = update_domain_params(center_real, center_imag, event.xdata, event.ydata, zoom_fac)
        frame = compute_frame(complex(center_real, center_imag), radius, res)
        ax.clear()
        ax.imshow(frame, cmap='viridis')
        plt.gcf().canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)

    frame = compute_frame(complex(center_real, center_imag), radius, res)
    ax.imshow(frame, cmap='viridis')
    ax.set_xticks([])
    ax.set_yticks([])

    while True:
        fig.canvas.draw()
        fig.canvas.flush_events()


def zoom_mandelbrot_auto(resolution: int = 200, start_radius: float = 2., start_real: float = 0,
                         start_imag: float = 0, zoom_fac: float = 1.05, center_update_freq: int = 20):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot()

    global radius, res
    res = resolution
    radius = start_radius
    center_real = start_real
    center_imag = start_imag

    frame = compute_frame(complex(center_real, center_imag), radius, res)
    ax.imshow(frame, cmap='viridis')
    ax.set_xticks([])
    ax.set_yticks([])

    center_imag_coords, center_real_coords = get_center_candidate_coords(frame)
    sampled_coord_idx = np.random.choice(np.arange(len(center_real_coords)))
    center_real_coord = center_real_coords[sampled_coord_idx]
    center_imag_coord = center_imag_coords[sampled_coord_idx]

    i = 0
    while True:
        center_real, center_imag = update_domain_params(center_real, center_imag, center_real_coord, center_imag_coord,
                                                        zoom_fac)
        frame = compute_frame(complex(center_real, center_imag), radius, res)
        ax.clear()
        ax.imshow(frame, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.draw()
        fig.canvas.flush_events()

        if i % center_update_freq == 0:
            center_imag_coords, center_real_coords = get_center_candidate_coords(frame)
            if len(center_real_coords) == 0:
                break
            center_imag_coord, center_real_coord = get_new_center_coords(center_imag_coord, center_imag_coords,
                                                                         center_real_coord, center_real_coords)
        else:
            center_real_coord = res / 2
            center_imag_coord = res / 2

        i += 1


def get_new_center_coords(center_imag_coord: float, center_imag_coords: np.ndarray, center_real_coord: float,
                          center_real_coords: np.ndarray) -> Tuple[float, float]:
    sampled_coord_idx = np.argmin(
        [np.sqrt((center_real - center_real_coord) ** 2 + (center_imag - center_imag_coord) ** 2)
         for center_real, center_imag in
         zip(center_real_coords, center_imag_coords)
         ]
    )
    center_real_coord = center_real_coords[sampled_coord_idx]
    center_imag_coord = center_imag_coords[sampled_coord_idx]
    return center_imag_coord, center_real_coord


class VisualizationMode(Enum):
    CLICK = 'click'
    AUTO = 'auto'


def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=VisualizationMode, default=VisualizationMode.AUTO)
    parser.add_argument('--resolution', type=int, default=200)
    parser.add_argument('--zoom_fac', type=float, default=1.05)
    args = parser.parse_args()

    if args.mode == VisualizationMode.AUTO:
        zoom_mandelbrot_auto(resolution=args.resolution, zoom_fac=args.zoom_fac)
    elif args.mode == VisualizationMode.CLICK:
        zoom_mandelbrot_click(resolution=args.resolution, zoom_fac=args.zoom_fac)
    else:
        raise NotImplementedError(f'Unsupported visualization mode {args.mode}')


if __name__ == '__main__':
    main()
