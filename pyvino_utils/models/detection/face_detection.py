import cv2

from ..openvino_base.base_model import Base


class FaceDetection(Base):
    """Class for the Face Detection Model."""

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

    def preprocess_output(self, inference_results, image, show_bbox=False):
        """Draw bounding boxes onto the Face Detection frame."""
        results = {}
        if not (self._init_image_w and self._init_image_h):
            raise RuntimeError("Initial image width and height cannot be None.")
        if len(inference_results) == 1:
            inference_results = inference_results[0]

        bbox_coord = []
        for box in inference_results[0][0]:  # Output shape is 1x1xNx7
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * self._init_image_w)
                ymin = int(box[4] * self._init_image_h)
                xmax = int(box[5] * self._init_image_w)
                ymax = int(box[6] * self._init_image_h)
                bbox_coord.append((xmin, ymin, xmax, ymax))
                if show_bbox:
                    self.draw_output(image, xmin, ymin, xmax, ymax)
        results['image'] = image
        results["bbox_coord"] = bbox_coord
        return results

    @staticmethod
    def draw_output(
        image,
        xmin,
        ymin,
        xmax,
        ymax,
        label="Face",
        bbox_color=(0, 255, 0),
        padding_size=(0.05, 0.25),
        text_color=(255, 255, 255),
        text_scale=2,
        text_thickness=2,
    ):

        cv2.rectangle(
            image, (xmin, ymin), (xmax, ymax,), color=bbox_color, thickness=2,
        )

        ((label_width, label_height), _) = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, fontScale=text_scale, thickness=text_thickness,
        )

        cv2.rectangle(
            image,
            (xmin, ymin),
            (
                int(xmin + label_width + label_width * padding_size[0]),
                int(ymin + label_height + label_height * padding_size[1]),
            ),
            color=bbox_color,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            image,
            label,
            org=(xmin, int(ymin + label_height + label_height * padding_size[1]),),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=text_scale,
            color=text_color,
            thickness=text_thickness,
        )
