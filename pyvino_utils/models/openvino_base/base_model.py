import os
import time
from abc import ABC, abstractmethod, abstractstaticmethod
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from openvino.inference_engine import IECore, IENetwork, get_version


def openvino_version_check():
    version = tuple(map(int, get_version().split(".")))[:2]
    if version != (2, 1):
        logger.warning(
            f"OpenVINO version: {version!r} not compatible with this library, "
            f"expected version: 2.1.xxx"
        )


class InvalidModel(Exception):
    pass


class Base(ABC):
    """Model Base Class"""

    def __init__(
        self,
        model_name,
        source_width=None,
        source_height=None,
        device="CPU",
        threshold=0.60,
        extensions=None,
    ):
        self.model_weights = f"{model_name}.bin"
        self.model_structure = f"{model_name}.xml"
        assert (
            Path(self.model_weights).absolute().exists()
            and Path(self.model_structure).absolute().exists()
        )

        openvino_version_check()

        self.device = device
        self.threshold = threshold
        self._model_size = os.stat(self.model_weights).st_size / 1024.0 ** 2

        self._ie_core = IECore()
        self.model = self._get_model()

        # Get the input layer
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self._init_image_w = source_width
        self._init_image_h = source_height
        self.exec_network = None
        self.perf_stats = {}
        self.load_model()

    @property
    def model_size(self):
        if hasattr(self, "model_weights"):
            return os.stat(self.model_weights).st_size / 1024.0 ** 2

    def _get_model(self):
        """Helper function for reading the network."""
        try:
            try:
                model = self._ie_core.read_network(
                    model=self.model_structure, weights=self.model_weights
                )
            except AttributeError:
                logger.warn("Using an old version of OpenVINO, consider updating it!")
                model = IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception:
            raise ValueError(
                "Could not Initialise the network. "
                "Have you entered the correct model path?"
            )
        else:
            return model

    def load_model(self):
        """Load the model into the plugin"""
        if self.exec_network is None:
            start_time = time.time()
            self.exec_network = self._ie_core.load_network(
                network=self.model, device_name=self.device
            )
            self._model_load_time = (time.time() - start_time) * 1000
            logger.info(
                f"Model: {self.model_structure} took "
                f"{self._model_load_time:.3f} ms to load."
            )

    def preprocess_input(self, image, height=None, width=None, **kwargs):
        """Helper function for processing frame"""
        if (height and width) is None:
            height, width = self.input_shape[2:]

        def transpose_image(f):
            f = f.transpose((2, 0, 1))
            f = f.reshape(1, *f.shape)
            return f

        gray_p_frame = None
        p_frame = cv2.resize(image, (width, height))
        # Change data layout from HWC to CHW
        p_frame = transpose_image(p_frame)

        if kwargs["gray_enabled"]:
            gray_p_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_p_frame = cv2.GaussianBlur(gray_p_frame, (5, 5), 0)

        return p_frame, gray_p_frame

    def predict(self, image, request_id=0, show_bbox=False, **kwargs):
        if not isinstance(image, np.ndarray):
            raise IOError("Image not parsed correctly.")

        p_image, gray_p_frame = self.preprocess_input(image, **kwargs)

        predict_start_time = time.time()
        self.exec_network.start_async(
            request_id=request_id, inputs={self.input_name: p_image}
        )
        status = self.exec_network.requests[request_id].wait(-1)
        if status == 0:
            pred_result = []
            for output_name, data_ptr in self.model.outputs.items():
                pred_result.append(
                    self.exec_network.requests[request_id].outputs[output_name]
                )
            self.perf_stats[output_name] = self.exec_network.requests[
                request_id
            ].get_perf_counts()
            predict_end_time = float(time.time() - predict_start_time) * 1000
            bbox, _ = self.preprocess_output(
                pred_result, image, show_bbox=show_bbox, **kwargs
            )
            return (predict_end_time, bbox, gray_p_frame)

    @staticmethod
    @abstractstaticmethod
    def draw_output(image):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def preprocess_output(self, inference_results, image, show_bbox=False, **kwargs):
        """Draw bounding boxes onto the frame."""
        raise NotImplementedError("Please Implement this method")
