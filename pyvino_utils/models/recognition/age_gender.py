import time

import cv2
import numpy as np

from ..openvino_base.base_model import Base

GENDER = ["Female", "Male"]


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
        **kwargs,
    ):
        super().__init__(
            model_name,
            source_width,
            source_height,
            device,
            threshold,
            extensions,
            **kwargs,
        )

    def preprocess_output(self, inference_results, image, show_bbox, **kwargs):
        results = {}
        age_conv3 = np.squeeze(inference_results[0])
        gender_prob = np.squeeze(inference_results[1])

        results["gender"] = GENDER[np.argmax(gender_prob)]
        results["age"] = int(np.round(age_conv3 * 100))
        if show_bbox:
            self.draw_output(results, image)
        return results, image

    @staticmethod
    def draw_output(results, image, **kwargs):
        w, h = 15, abs(image.shape[0] - 100)
        cv2.putText(
            image,
            f"Age: {results['age']}, Gender: {results['gender']}",
            (w, h),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
