import time

import cv2
import numpy as np

from ..openvino_base.base_model import Base


class GazeEstimation(Base):
    """Class for the Gaze Estimation Recognition Model."""

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
        gaze_vector = dict(zip(["x", "y", "z"], np.vstack(inference_results).ravel()))

        # TODO: Figure out why I had to comment this code out?
        # roll_val = kwargs["head_pose_angles"]["roll"]

        # cos_theta = math.cos(roll_val * math.pi / 180)
        # sin_theta = math.sin(roll_val * math.pi / 180)

        # coords = {"x": None, "y": None}
        # coords["x"] = gaze_vector["x"] * cos_theta + gaze_vector["y"] * sin_theta
        # coords["y"] = gaze_vector["y"] * cos_theta - gaze_vector["x"] * sin_theta
        if show_bbox:
            self.draw_output(gaze_vector, image, **kwargs)

        results["Gaze_Vector"] = gaze_vector

        return results, image

    @staticmethod
    def draw_output(coords, image, **kwargs):
        left_eye_point = kwargs["eyes_coords"]["left_eye_point"]
        right_eye_point = kwargs["eyes_coords"]["right_eye_point"]
        cv2.arrowedLine(
            image,
            (
                left_eye_point[0] + int(coords["x"] * 500),
                left_eye_point[1] - int(coords["y"] * 500),
            ),
            (left_eye_point[0], left_eye_point[1]),
            color=(0, 0, 255),
            thickness=2,
            tipLength=0.2,
        )
        cv2.arrowedLine(
            image,
            (
                right_eye_point[0] + int(coords["x"] * 500),
                right_eye_point[1] - int(coords["y"] * 500),
            ),
            (right_eye_point[0], right_eye_point[1]),
            color=(0, 0, 255),
            thickness=2,
            tipLength=0.2,
        )

    @staticmethod
    def show_text(
        image, coords, pos=550, font_scale=1.5, color=(255, 255, 255), thickness=1
    ):
        """Helper function for showing the text on frame."""
        height, _ = image.shape[:2]
        ypos = abs(height - pos)
        text = "Gaze Vector: " + ", ".join(f"{x}: {y:.2f}" for x, y in coords.items())
        cv2.putText(
            image,
            text,
            (15, ypos),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
        )

    def preprocess_input(self, image, **kwargs):
        width, height = self.model.inputs["left_eye_image"].shape[2:]

        p_left_eye_image = Base.preprocess_input(
            Base, kwargs["eyes_coords"]["left_eye_image"], width, height
        )
        p_right_eye_image = Base.preprocess_input(
            Base, kwargs["eyes_coords"]["right_eye_image"], width, height
        )

        return p_left_eye_image, p_right_eye_image

    def predict(self, image, request_id=0, show_bbox=False, **kwargs):
        p_left_eye_image, p_right_eye_image = self.preprocess_input(image, **kwargs)
        head_pose_angles = list(kwargs.get("head_pose_angles").values())

        predict_start_time = time.time()
        status = self.exec_network.start_async(
            request_id=request_id,
            inputs={
                "left_eye_image": p_left_eye_image,
                "right_eye_image": p_right_eye_image,
                "head_pose_angles": head_pose_angles,
            },
        )
        status = self.exec_network.requests[request_id].wait(-1)

        if status == 0:
            pred_result = []
            for output_name, data_ptr in self.model.outputs.items():
                pred_result.append(
                    self.exec_network.requests[request_id].outputs[output_name]
                )
            predict_end_time = float(time.time() - predict_start_time) * 1000
            gaze_vector, _ = self.preprocess_output(
                pred_result, image, show_bbox=show_bbox, **kwargs
            )
        return (predict_end_time, gaze_vector)
