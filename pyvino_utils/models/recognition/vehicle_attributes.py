import time

import cv2
import numpy as np

from ..openvino_base.base_model import Base

CAR_COLORS = ["white", "gray", "yellow", "red", "green", "blue", "black"]
CAR_TYPES = ["car", "bus", "truck", "van"]


class VehicleAttrs(Base):
    """Class for the Vehicle Attributes Recognition Model."""

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
        """
        Handles the output of the Car Metadata model.
        """
        results = {}
        color = inference_results.get("color")
        output_type = inference_results.get("type")
        # Get the argmax of the "color" output
        results["color"] = np.argmax(color)
        # Get the argmax of the "type" output
        output_type = np.argmax(output_type)
        results["output_type"] = output_type
        if show_bbox:
            self.draw_output(image, results)
        return results

    @staticmethod
    def draw_output(image, results, **kwargs):
        pass
