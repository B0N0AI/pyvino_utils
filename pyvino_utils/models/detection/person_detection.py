import cv2  # noqa: F401
import numpy as np  # noqa: F401

from ..openvino_base.base_model import Base


class PersonDetection(Base):
    """Class for the Person Detection Model."""

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

    def preprocess_output(self, inference_results, image, show_bbox=False, **kwargs):
        raise NotImplementedError("Method not implemented!")

    @staticmethod
    def draw_output(coords, image, **kwargs):
        raise NotImplementedError("Method not implemented!")
