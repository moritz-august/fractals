from enum import Enum

import numpy as np


class VisualizationMode(Enum):
    CLICK = 'click'
    AUTO = 'auto'


def normalize_min_max(frame: np.ndarray) -> np.ndarray:
    frame -= frame.min()
    frame /= frame.max()

    return frame
