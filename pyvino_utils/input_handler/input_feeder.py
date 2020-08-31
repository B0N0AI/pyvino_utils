import mimetypes
import os

import cv2
from tqdm import tqdm

from loguru import logger
from numpy import ndarray

__all__ = ['InputFeeder']

class FormatNotSupported(Exception):
    pass


class InputFeeder:
    def __init__(self, input_feed=None, cam_input=0):
        """
        This class can be used to feed input from an image, webcam, or video to your model.

        Parameters
        ----------
        input_feed: str
            The file that contains the input image or video file.
            Leave empty for cam input_type.
        cam_input: int
            WebCam input [Default: 0]

        Example
        -------
        ```
            feed = InputFeeder(input_feed='video.mp4')
            for frame in feed.next_frame():
                do_something(frame)
            feed.close()
        ```
        """
        self.input_feed = input_feed
        assert isinstance(self.input_feed, str)
        self.check_file_exists(self.input_feed)
        try:
            self._input_type, _ = mimetypes.guess_type(self.input_feed)
            assert isinstance(self._input_type, str)
        except AssertionError:
            self._input_type = ""
        self._progress_bar = None
        self.load_feed(cam_input)

    def load_feed(self, cam_input):
        if "video" in self._input_type:
            self.cap = cv2.VideoCapture(self.input_feed)
        elif "image" in self._input_type:
            self.cap = cv2.imread(self.input_feed)
        elif "cam" in self.input_feed.lower():
            self._input_type = self.input_feed
            self.cap = cv2.VideoCapture(cam_input)
        else:
            msg = f"Source: {self.input_feed} not supported!"
            logger.warn(msg)
            raise FormatNotSupported(msg)
        logger.info(f"Loaded input source type: {self._input_type}")

    @staticmethod
    def check_file_exists(file):
        if "cam" in file:
            return

        if not os.path.exists(os.path.abspath(file)):
            raise FileNotFoundError(f"{file} does not exist.")

    @property
    def source_width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def source_height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def video_len(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def frame_size(self):
        return (self.source_height, self.source_width)

    @property
    def progress_bar(self):
        if not self._progress_bar:
            self._progress_bar = tqdm(total=int(self.video_len - self.fps + 1))
        return self._progress_bar

    def resize(self, frame, height=None, width=None):
        if (height and width) is None:
            width, height = (self.source_width // 2, self.source_height // 2)
        return cv2.resize(frame, (width, height))

    def draw_center_cirle(self, frame, radius=100, color=(0, 255, 0), thickness=5):
        cv2.circle(
            frame,
            (self.source_width // 2, self.source_height // 2,),
            radius,
            color,
            thickness,
        )

    def show(self, frame, frame_name="video"):
        cv2.imshow(frame_name, frame)

    def write_video(self, output_path=".", filename="output_video.mp4"):
        out_video = cv2.VideoWriter(
            os.path.join(output_path, filename),
            cv2.VideoWriter_fourcc(*"avc1"),
            fps,
            (self.source_width, self.source_height),
            True,
        )
        return out_video

    def next_frame(self, quit_key="q", progress=True):
        """Returns the next image from either a video file or webcam."""
        while self.cap.isOpened():
            if progress:
                self.progress_bar.update(1)
            flag = False
            for _ in range(1):
                flag, frame = self.cap.read()

            if not flag:
                break
            yield frame

            key = cv2.waitKey(1) & 0xFF
            # if `quit_key` was pressed, break from the loop
            if key == ord(quit_key):
                break

    # TODO: Add context-manager to handle the closing
    def close(self):
        """Closes the VideoCapture."""
        if "image" in self._input_type:
            self.cap.release()
        if self.progress_bar:
            self.progress_bar.close()
        cv2.destroyAllWindows()
        logger.info("============ CleanUp! ============")
