import os
import time
from abc import ABC, abstractmethod, abstractstaticmethod
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from openvino.inference_engine import IECore, IENetwork, get_version

from .faults import InvalidImageArray, InvalidModel


def openvino_version_check():
    version = tuple(map(int, get_version().split(".")))[:2]
    if version != (2, 1):
        logger.warning(
            f"OpenVINO version: {version!r} not compatible with this library, "
            f"expected version: 2.1.xxx"
        )


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
        **kwargs,
    ):
        self.model_weights = f"{model_name}.bin"
        self.model_structure = f"{model_name}.xml"
        assert (
            Path(self.model_weights).absolute().exists()
            and Path(self.model_structure).absolute().exists()
        )

        openvino_version_check()

        self.threshold = threshold
        self._device = device
        self._model_size = os.stat(self.model_weights).st_size / 1024.0 ** 2

        self._ie_core = IECore()
        self.model = self._get_model()

        # Get the input layer
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self._update_source_resolution(source_width, source_height, **kwargs)
        self.exec_network = None
        self.perf_stats = {}
        self.load_model()

    def _update_source_resolution(self, source_width, source_height, **kwargs):
        if kwargs.get("input_feed"):
            self._init_image_h = kwargs.get("input_feed").source_height
            self._init_image_w = kwargs.get("input_feed").source_width
        else:
            self._init_image_w = source_width
            self._init_image_h = source_height

    @property
    def model_size(self):
        """Get the size of model in Megabytes."""
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
                logger.warn(
                    f"Using an old version of OpenVINO, "
                    f"Please update it to version: {get_version()}!"
                )
                model = IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception:
            msg = (
                "Could not Initialise the network. "
                "Have you entered the correct model path?"
            )
            logger.exception(msg)
            raise InvalidModel(msg)
        else:
            return model

    def load_model(self):
        """Load the model into the plugin."""
        if self.exec_network is None:
            start_time = time.time()
            self.exec_network = self._ie_core.load_network(
                network=self.model, device_name=self._device
            )
            self._model_load_time = (time.time() - start_time) * 1000
            logger.info(
                f"Model: {self.model_structure} took "
                f"{self._model_load_time:.3f} ms to load."
            )
            self._check_supported_layers()

    def _check_supported_layers(self):
        """Check if layers are supported by the device."""
        if self.exec_network is None:
            supported_layers = self.ie_core.query_network(
                network=self.exec_network, device_name=self._device
            )

            unsupported_layers = [
                layer
                for layer in self.exec_network.layers.keys()
                if layer not in supported_layers
            ]
            if len(unsupported_layers) != 0:
                logger.warning(
                    f"Unsupported layers found: {unsupported_layers}, "
                    "Check whether extensions are available to add to IECore."
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

        if kwargs.get("gray_enabled"):
            gray_p_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_p_frame = cv2.GaussianBlur(gray_p_frame, (5, 5), 0)

        return p_frame, gray_p_frame

    def predict(self, image, request_id=0, show_bbox=False, **kwargs):
        if not isinstance(image, np.ndarray):
            raise InvalidImageArray("Image not parsed correctly.")

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
    def draw_output(results, image):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def preprocess_output(self, inference_results, image, show_bbox=False, **kwargs):
        """Draw bounding boxes onto the frame.

        Returns:
        --------

        results: dict
            inference results
        image: np.ndarray
            image array
        """
        raise NotImplementedError("Please Implement this method")
