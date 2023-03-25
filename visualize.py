from typing import Optional
from typing import Tuple

import cv2
import numpy as np

from mandelbrot import compute_frame


class ZoomVisualizer:

    def __init__(self, resolution: int = 200, zoom_fac: float = 1.2,
                 start_radius: float = 2., start_real: float = 0, start_imag: float = 0, output_resolution: int = 800):
        self.resolution = resolution
        self.output_resolution = output_resolution
        self.radius = start_radius
        self.center_real = start_real
        self.center_imag = start_imag
        self.zoom_fac = zoom_fac

    def update_domain_params(self, real_coord: float, imag_coord: float, zoom_fac: Optional[float] = None):
        min_real = self.center_real - self.radius
        min_imag = self.center_imag - self.radius
        self.center_real = min_real + real_coord / self.resolution * 2 * self.radius
        self.center_imag = min_imag + imag_coord / self.resolution * 2 * self.radius

        if zoom_fac:
            self.radius /= zoom_fac
        else:
            self.radius /= self.zoom_fac

    def visualize_frame(self, frame: np.ndarray):
        frame_vis = cv2.applyColorMap((frame * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        frame_vis = cv2.resize(frame_vis, (self.output_resolution, self.output_resolution), cv2.INTER_LINEAR)
        cv2.imshow('Fractal Zoom', frame_vis)


class ZoomClickVisualizer(ZoomVisualizer):

    def __init__(self, resolution: int = 200, zoom_fac: float = 1.2, start_radius: float = 2., start_real: float = 0,
                 start_imag: float = 0):
        super(ZoomClickVisualizer, self).__init__(resolution, zoom_fac, start_radius, start_real, start_imag)

    def visualize(self):
        def mouse_callback(action, x, y, flags, *userdata):
            if action == cv2.EVENT_LBUTTONDOWN:
                self.update_domain_params(x, y)
                frame = compute_frame(complex(self.center_real, self.center_imag), self.radius, self.resolution)
                self.visualize_frame(frame)

        cv2.namedWindow('Fractal Zoom')
        cv2.setMouseCallback('Fractal Zoom', mouse_callback)

        frame = compute_frame(complex(self.center_real, self.center_imag), self.radius, self.resolution)
        self.visualize_frame(frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class ZoomAutoVisualizer(ZoomVisualizer):

    def __init__(self, resolution: int = 200, start_radius: float = 2., start_real: float = 0,
                 start_imag: float = 0, zoom_fac: float = 1.05, center_update_freq: int = 20):
        super(ZoomAutoVisualizer, self).__init__(resolution, zoom_fac, start_radius, start_real, start_imag)
        self.center_update_freq = center_update_freq
        self.frame_refresh_ms = 20
        self.initial_zoom_fac = 1.01

    @staticmethod
    def get_new_center_coords(center_imag_coord: float, center_imag_coords: np.ndarray, center_real_coord: float,
                              center_real_coords: np.ndarray) -> Tuple[float, float]:
        distances = np.sqrt((center_real_coords - center_real_coord) ** 2 +
                            (center_imag_coords - center_imag_coord) ** 2)
        sampled_coord_idx = np.argmin(distances)

        center_real_coord = center_real_coords[sampled_coord_idx]
        center_imag_coord = center_imag_coords[sampled_coord_idx]
        return center_imag_coord, center_real_coord

    @staticmethod
    def get_center_candidate_coords(frame: np.ndarray, target_val: float = 0.4, tolerance: float = 0.1) -> np.ndarray:
        return np.where(np.abs(frame - target_val) < tolerance)

    def visualize(self):

        frame = compute_frame(complex(self.center_real, self.center_imag), self.radius, self.resolution)
        cv2.namedWindow('Fractal Zoom')
        self.visualize_frame(frame)
        cv2.waitKey(self.frame_refresh_ms)

        zoom_start_imag, zoom_start_real = self.get_zoom_start(frame)
        center_imag_coord, center_real_coord = self.move_to_zoom_start(zoom_start_imag, zoom_start_real)

        i = 0
        while True:
            self.update_domain_params(center_real_coord, center_imag_coord)
            frame = compute_frame(complex(self.center_real, self.center_imag), self.radius, self.resolution)
            self.visualize_frame(frame)
            cv2.waitKey(self.frame_refresh_ms)

            if i > 0 and i % self.center_update_freq == 0:
                center_imag_coords, center_real_coords = self.get_center_candidate_coords(frame)
                if len(center_real_coords) == 0:
                    break
                center_imag_coord, center_real_coord = self.get_new_center_coords(center_imag_coord, center_imag_coords,
                                                                                  center_real_coord, center_real_coords)
            else:
                center_real_coord = self.resolution // 2
                center_imag_coord = self.resolution // 2

            i += 1

        cv2.destroyAllWindows()

    def move_to_zoom_start(self, zoom_start_imag: float, zoom_start_real: float) -> Tuple[int, int]:

        real_dist = zoom_start_real - self.center_real
        imag_dist = zoom_start_imag - self.center_imag
        real_direction = np.sign(real_dist)
        imag_direction = np.sign(imag_dist)
        real_update_step = np.maximum(np.abs(real_dist // imag_dist), 1)
        imag_update_step = np.maximum(np.abs(imag_dist // real_dist), 1)

        while True:
            center_real_coord = self.resolution // 2 + real_direction * real_update_step
            center_imag_coord = self.resolution // 2 + imag_direction * imag_update_step
            self.update_domain_params(center_real_coord, center_imag_coord, self.initial_zoom_fac)
            if (zoom_start_real - self.center_real) * real_dist < 0 and (
                    zoom_start_imag - self.center_imag) * imag_direction < 0:
                break
            frame = compute_frame(complex(self.center_real, self.center_imag), self.radius, self.resolution)
            self.visualize_frame(frame)
            cv2.waitKey(self.frame_refresh_ms)

        return center_imag_coord, center_real_coord

    def get_zoom_start(self, frame: np.ndarray) -> Tuple[float, float]:
        center_imag_coords, center_real_coords = self.get_center_candidate_coords(frame)
        sampled_coord_idx = np.random.choice(np.arange(len(center_real_coords)))
        zoom_start_real = center_real_coords[sampled_coord_idx]
        zoom_start_imag = center_imag_coords[sampled_coord_idx]
        min_real = self.center_real - self.radius
        min_imag = self.center_imag - self.radius
        zoom_start_real = min_real + zoom_start_real / self.resolution * 2 * self.radius
        zoom_start_imag = min_imag + zoom_start_imag / self.resolution * 2 * self.radius

        return zoom_start_imag, zoom_start_real
