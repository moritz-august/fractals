from typing import Tuple

import cv2
import numpy as np

from mandelbrot import compute_frame


class ZoomVisualizer:

    def __init__(self, resolution: int = 200, zoom_fac: float = 1.2, start_radius: float = 2., start_real: float = 0,
                 start_imag: float = 0):
        self.resolution = resolution
        self.radius = start_radius
        self.center_real = start_real
        self.center_imag = start_imag
        self.zoom_fac = zoom_fac

    def update_domain_params(self, real_coord: float, imag_coord: float):
        min_real = self.center_real - self.radius
        min_imag = self.center_imag - self.radius
        self.center_real = min_real + real_coord / self.resolution * 2 * self.radius
        self.center_imag = min_imag + imag_coord / self.resolution * 2 * self.radius
        self.radius /= self.zoom_fac


class ZoomClickVisualizer(ZoomVisualizer):

    def __init__(self, resolution: int = 200, zoom_fac: float = 1.2, start_radius: float = 2., start_real: float = 0,
                 start_imag: float = 0):
        super(ZoomClickVisualizer, self).__init__(resolution, zoom_fac, start_radius, start_real, start_imag)

    def visualize(self):
        def mouse_callback(action, x, y, flags, *userdata):
            if action == cv2.EVENT_LBUTTONDOWN:
                self.update_domain_params(x, y)
                frame = compute_frame(complex(self.center_real, self.center_imag), self.radius, self.resolution)
                print(x, y)
                cv2.imshow('Fractal Zoom', frame)

        cv2.namedWindow('Fractal Zoom')
        cv2.setMouseCallback('Fractal Zoom', mouse_callback)

        frame = compute_frame(complex(self.center_real, self.center_imag), self.radius, self.resolution)

        cv2.imshow('Fractal Zoom', frame)
        cv2.waitKey(0)
        cv2.destroyAllwindows()


class ZoomAutoVisualizer(ZoomVisualizer):

    def __init__(self, resolution: int = 200, start_radius: float = 2., start_real: float = 0,
                 start_imag: float = 0, zoom_fac: float = 1.05, center_update_freq: int = 20):
        super(ZoomAutoVisualizer, self).__init__(resolution, zoom_fac, start_radius, start_real, start_imag)
        self.center_update_freq = center_update_freq
        self.frame_refresh_ms = 20

    @staticmethod
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

    @staticmethod
    def get_center_candidate_coords(frame: np.ndarray, target_val: float = 0.5, tolerance: float = 0.1) -> np.ndarray:
        return np.where(np.abs(frame - target_val) < tolerance)

    def visualize(self):

        frame = compute_frame(complex(self.center_real, self.center_imag), self.radius, self.resolution)
        cv2.namedWindow('Fractal Zoom')
        cv2.imshow('Fractal Zoom', frame)
        cv2.waitKey(self.frame_refresh_ms)

        center_imag_coords, center_real_coords = self.get_center_candidate_coords(frame)
        sampled_coord_idx = np.random.choice(np.arange(len(center_real_coords)))
        center_real_coord = center_real_coords[sampled_coord_idx]
        center_imag_coord = center_imag_coords[sampled_coord_idx]

        i = 0
        while True:
            self.update_domain_params(center_real_coord, center_imag_coord)
            frame = compute_frame(complex(self.center_real, self.center_imag), self.radius, self.resolution)
            cv2.imshow('Fractal Zoom', frame)
            cv2.waitKey(self.frame_refresh_ms)

            if i % self.center_update_freq == 0:
                center_imag_coords, center_real_coords = self.get_center_candidate_coords(frame)
                if len(center_real_coords) == 0:
                    break
                center_imag_coord, center_real_coord = self.get_new_center_coords(center_imag_coord, center_imag_coords,
                                                                                  center_real_coord, center_real_coords)
            else:
                center_real_coord = self.resolution / 2
                center_imag_coord = self.resolution / 2

            i += 1

        cv2.destroyAllwindows()
