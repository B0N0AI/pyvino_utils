import time

import cv2
import numpy as np

from ..openvino_base.base_model import Base


class TextRecognition(Base):
    """Class for the Text Recognition Model."""

    def __init__(
        self,
        model_name,
        source_width=None,
        source_height=None,
        device="CPU",
        threshold=0.60,
        extensions=None,
        **kwargs
    ):
        super().__init__(
            model_name,
            source_width,
            source_height,
            device,
            threshold,
            extensions,
            **kwargs
        )

    def preprocess_output(self, inference_results, image, show_bbox, **kwargs):
        results = {}
        # Do stuff
        results['image'] = image
        return results

    @staticmethod
    def draw_output(results, image, **kwargs):
        pass
