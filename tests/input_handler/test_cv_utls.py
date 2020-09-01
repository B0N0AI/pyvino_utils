import unittest

from pyvino_utils.opencv_utils import cv_utils


class test_cv_utils(unittest.TestCase):
    def setUp(self):
        self.DUT = cv_utils

    def test_select_color(self):
        self.assertEqual(self.DUT.select_color("blue"), (255, 0, 0))
        self.assertEqual(self.DUT.select_color("yellow"), (0, 255, 0))
        with self.assertRaises(AttributeError):
            self.DUT.select_color((255, 255, 0))
