import time

import cv2
import numpy as np

from ..openvino_base.base_model import Base


class AgeGender(Base):
    """Class for the Age Gender Recognition Model."""

    def __init__(
        self,
        model_name,
        source_width=None,
        source_height=None,
        device="CPU",
        threshold=0.60,
        extensions=None,
    ):
        super().__init__(
            model_name, source_width, source_height, device, threshold, extensions,
        )

    def preprocess_output(self, inference_results, image, show_bbox, **kwargs):
        pass

    @staticmethod
    def draw_output(coords, image, **kwargs):
        pass
