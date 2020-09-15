import math

import cv2
import numpy as np

from ..openvino_base.base_model import Base, InvalidModel


class HumanPoseEstimation(Base):
    """Class for the Human Pose Estimation Model."""

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
        """
        Handles the output of the Pose Estimation model.
        """
        results = {}
        # FIXME: untested logic
        # if kwargs.get("heatmaps"):
        #     # Extract only the second blob output (keypoint heatmaps)
        #     heatmaps = inference_results.get("Mconv7_stage2_L2")
        #     # Resize the heatmap back to the size of the input
        #     input_shape = image.shape[:2]
        #     out_heatmap = np.zeros([heatmaps.shape[1], input_shape[0], input_shape[1]])
        #     results["heatmaps"] = out_heatmap
        #     if show_bbox:
        #         self.draw_output(results, image)
        return results

    @staticmethod
    def draw_output(results, image):
        pass
        # FIXME: untested logic
        # if results.get("heatmaps"):
        #     input_shape = image.shape[:2]
        #     for idx, _ in enumerate(heatmaps[0]):
        #         cv2.resize(heatmaps[0][idx], input_shape[0:2][::-1])
