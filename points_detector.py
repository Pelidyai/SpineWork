import cv2
import numpy as np

from verts_models.merged_model import MergedModel
from mask_creation import get_opening


class PointsDetector:
    def __init__(self):
        self.model = MergedModel()

    def get_points(self, images):
        result = []
        i = 0
        for img in images:
            if img.shape[0] != 0 and img.shape[1] != 0:
                result.append(self.model.predict(get_opening(img), img))
            i += 1
        return result
