#!/usr/bin/env python3

import argparse

from pyvino_utils import InputFeeder
from pyvino_utils.models.recognition import age_gender


def arg_parser():
    parser = argparse.ArgumentParser(
        description="A simple OpenVINO based Age and Gender Recognition running on CPU."
    )
    parser.add_argument("-i", "--input", help="Video or Image input.", required=True)
    parser.add_argument(
        "-m", "--model", help="model name (without an extension).", required=True
    )
    parser.add_argument(
        "-b", "--show-bbox", action="store_true", help="Show bounding box."
    )
    return parser.parse_args()


def main(args):
    input_feed = InputFeeder(input_feed=args.input)
    age_gender_detector = age_gender.AgeGender(
        model_name=args.model, input_feed=input_feed
    )

    for frame in input_feed.next_frame(progress=False):
        inference_results = age_gender_detector.predict(frame, show_bbox=args.show_bbox)
        # Do stuff with inference_results.
        if args.show_bbox:
            input_feed.show(frame)
    input_feed.close()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
