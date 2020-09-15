import argparse

from pyvino_utils import InputFeeder
from pyvino_utils.models.detection import face_detection


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Video or image input.", required=True)
    parser.add_argument(
        "-m", "--model", help="Face detection model name (no extension)", required=True
    )
    parser.add_argument("-b", "--show-bbox", help="show bonding box")
    return parser.args


def main(args):
    video_feed = InputFeeder(input_feed=args.input)
    face_detector = face_detection.FaceDetection(
        model_name=args.model, video_feed=video_feed
    )

    for frame in video_feed.next():
        predict_time, face_bboxes = face_detector.predict(frame, show_bbox=args.show_bbox)
    video_feed.close()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
