import cv2
import numpy as np

from ..openvino_base.base_model import Base


class FacialLandmarks(Base):
    """Class for the Facial Landmarks Recognition Model."""

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
        self._model_type = (
            "landmarks-regression-retail"
            if "regression" in model_name
            else "facial-landmarks-35-adas"
        )
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
        """Draw bounding boxes onto the Facial Landmarks frame."""
        flattened_predictions = np.vstack(inference_results).ravel()
        results = {}
        face_h, face_w = image.shape[:2]

        if len(flattened_predictions) == 70:
            landmarks = []
            for i in range(flattened_predictions.size // 2):
                x_coord = int(face_w * flattened_predictions[2 * i])
                y_coord = int(face_h * flattened_predictions[2 * i + 1])
                landmarks.append((x_coord, y_coord))

            face_landmarks = {
                "type": self._model_type,
                "eyes_coords": landmarks[:4],
                "nose_coords": landmarks[4:8],
                "mouth_coords": landmarks[8:12],
                "face_contour": landmarks[12:],
            }
        else:
            coord_mapping = dict(
                zip(
                    (
                        "left_eye_x_coord",
                        "left_eye_y_coord",
                        "right_eye_x_coord",
                        "right_eye_y_coord",
                        "nose_coord",
                        "left_mouth_coord",
                        "right_mouth_coord",
                    ),
                    flattened_predictions,
                )
            )

            def get_eye_points(eye_coords, eye_size=10):

                eye_min = eye_coords - eye_size
                eye_max = eye_coords + eye_size
                return map(int, [eye_coords, eye_min, eye_max])

            # left eye offset of face
            left_eye_x_coord, left_eye_xmin, left_eye_xmax = get_eye_points(
                coord_mapping["left_eye_x_coord"] * face_w
            )

            # left eye offset of face
            left_eye_y_coord, left_eye_ymin, left_eye_ymax = get_eye_points(
                coord_mapping["left_eye_y_coord"] * face_h
            )

            # right eye offset of face
            right_eye_x_coord, right_eye_xmin, right_eye_xmax = get_eye_points(
                coord_mapping["right_eye_x_coord"] * face_w
            )

            # right eye offset of face
            right_eye_y_coord, right_eye_ymin, right_eye_ymax = get_eye_points(
                coord_mapping["right_eye_y_coord"] * face_h
            )
            eye_size = 10

            left_eye_x_coord = int(flattened_predictions[0] * face_w)
            # left eye offset of face
            left_eye_xmin = left_eye_x_coord - eye_size
            left_eye_xmax = left_eye_x_coord + eye_size

            left_eye_y_coord = int(flattened_predictions[1] * face_h)
            # left eye offset of face
            left_eye_ymin = left_eye_y_coord - eye_size
            left_eye_ymax = left_eye_y_coord + eye_size

            right_eye_x_coord = int(flattened_predictions[2] * face_w)
            # right eye offset of face
            right_eye_xmin = right_eye_x_coord - eye_size
            right_eye_xmax = right_eye_x_coord + eye_size

            right_eye_y_coord = int(flattened_predictions[3] * face_h)
            # right eye offset of face
            right_eye_ymin = right_eye_y_coord - eye_size
            right_eye_ymax = right_eye_y_coord + eye_size
            # nose coordinates
            nose_coord = coord_mapping["nose_coord"]

            # mouth coordinates
            left_part_mouth = coord_mapping["left_mouth_coord"] * face_w
            right_part_mouth = coord_mapping["right_mouth_coord"] * face_w

            face_landmarks = {
                "type": self._model_type,
                "eyes_coords": {
                    "left_eye_point": (left_eye_x_coord, left_eye_y_coord),
                    "right_eye_point": (right_eye_x_coord, right_eye_y_coord),
                    "left_eye_image": image[
                        left_eye_ymin:left_eye_ymax, left_eye_xmin:left_eye_xmax
                    ],
                    "right_eye_image": image[
                        right_eye_ymin:right_eye_ymax, right_eye_xmin:right_eye_xmax,
                    ],
                },
                "nose_coords": {"nose_coords": nose_coord},
                "mouth_coords": {"mouth_coords": [left_part_mouth, right_part_mouth]},
            }

        results["face_landmarks"] = face_landmarks
        results["image"] = image
        if show_bbox:
            self.draw_output(results, image, **kwargs)
        return results

    @staticmethod
    def draw_output(results, image, radius=20, color=(0, 0, 255), thickness=2, **kwargs):
        """Draw a circle around ROI"""
        pass
        # TODO: Fix this for the handle the 2 different types of landmarks.
        # for landmark in face_landmarks:
        #     if landmark == "eyes_coords":
        #         for eye, coords in face_landmarks["eyes_coords"].items():
        #             if "point" in eye:
        #                 cv2.circle(
        #                     image, (coords[0], coords[1]), radius, color, thickness,
        #                 )
