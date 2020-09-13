import time

import cv2
import numpy as np

from ..openvino_base.base_model import Base


EMOTION_STATES = ("neutral", "happy", "sad", "surprise", "anger")


class Emotions(Base):
    """Class for the Emotions Recognition Model."""

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
        results = {}
        emo_state = np.vstack(inference_results).ravel()
        results["emotional_state"] = EMOTION_STATES[np.argmax(emo_state)]

        if show_bbox:
            self.draw_output(results, image)

        return results, image

    @staticmethod
    def draw_output(results, image, **kwargs):

        cv2.putText(
            image,
            f"Emotional State: {results['emotional_state']}",
            org=(image.shape[1]//4, image.shape[0]//2),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2,
            color=(0,0,255),
            thickness=2,
        )
