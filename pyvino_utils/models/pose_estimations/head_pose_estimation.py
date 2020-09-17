import math

import cv2
import numpy as np

from ..openvino_base.base_model import Base, InvalidModel


class HeadPoseEstimation(Base):
    """Class for the Head Pose Estimation Model."""

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

    def preprocess_output(self, inference_results, image, show_bbox=False, **kwargs):
        """
        Estimate the Head Pose on a cropped face.

        Example
        -------
        Model: head-pose-estimation-adas-0001

            Output layer names in Inference Engine format:

            name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
            name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
            name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).

        """
        results = {}
        if len(inference_results) != 3:
            msg = (
                f"The model:{self.model_structure} does not contain expected output "
                "shape as per the docs."
            )
            self.logger.error(msg)
            raise InvalidModel(msg)

        output_layer_names = ["yaw", "pitch", "roll"]
        flattened_predictions = np.vstack(inference_results).ravel()
        results["head_pose_angles"] = dict(zip(output_layer_names, flattened_predictions))
        if show_bbox:
            self.draw_output(results, image, **kwargs)
        results["image"] = image
        return results

    @staticmethod
    def draw_output(results, image, **kwargs):
        """Draw head pose estimation on frame.

        Ref:
        https://github.com/natanielruiz/deep-head-pose/blob/master/code/utils.py#L86+L117
        """
        yaw, pitch, roll = results["head_pose_angles"].values()

        yaw = -(yaw * np.pi / 180)
        pitch = pitch * np.pi / 180
        roll = roll * np.pi / 180

        height, width = image.shape[:2]
        tdx = width / 2
        tdy = height / 2
        size = 1000

        # X-Axis pointing to right. drawn in red
        x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
        y1 = (
            size
            * (
                math.cos(pitch) * math.sin(roll)
                + math.cos(roll) * math.sin(pitch) * math.sin(yaw)
            )
            + tdy
        )

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
        y2 = -(
            size
            * (
                math.cos(pitch) * math.cos(roll)
                - math.sin(pitch) * math.sin(yaw) * math.sin(roll)
            )
            + tdy
        )

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (math.sin(yaw)) + tdx
        y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

        cv2.line(image, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
        cv2.line(image, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.line(image, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

        return image

    @staticmethod
    def show_text(
        image, coords, pos=500, font_scale=1.5, color=(255, 255, 255), thickness=1
    ):
        """Helper function for showing the text on frame."""
        height, _ = image.shape[:2]
        ypos = abs(height - pos)
        text = ", ".join(f"{x}: {y:.2f}" for x, y in coords.items())

        cv2.putText(
            image,
            text,
            (15, ypos),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
        )
