import unittest
from descrambler.descrambler import Scramble as Scramble
from os import listdir
from os.path import isfile, join
import cv2 as cv
import numpy as np


class TestDescrambling(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # loading all images from the testImages folder
        image_path = "../images/testImages"
        cls.dir_files = [f for f in listdir(image_path) if isfile(join(image_path, f))]
        image_files = []
        for i in range(len(cls.dir_files)):
            image_files.append(f"{image_path}/{cls.dir_files[i]}")

        cls.images = []
        for image in image_files:
            curr_image = cv.imread(image, cv.IMREAD_COLOR)
            height = curr_image.shape[0]
            width = curr_image.shape[1]
            rows = -1
            cols = -1
            # get the number of rows and cols
            for i in range(5, 11):
                if height % i == 0:
                    rows = i

            for x in range(5, 11):
                if width % x == 0:
                    cols = x

            if rows >= 5 and cols >= 5:
                cls.images.append((image, (rows, cols)))

    def test_img_num(self):
        self.assertEqual(len(self.dir_files), len(self.images), "Some images didn't pass the tests")


    def test_img_cutting(self):
        for image in self.images:
            # make a scramble object
            scramble = Scramble(image[0], image[1][0], image[1][1])
            scramble_image = scramble.get_image()

            original_img = cv.imread(image[0], cv.IMREAD_COLOR)
            print(f'TESTING: {image}')
            # unittest values: expected, actual
            self.assertEqual(original_img.shape, scramble_image.shape, f"Wrong size of image: {image[0]}")

            difference = cv.subtract(scramble_image, original_img)
            b, g, r = cv.split(difference)
            self.assertEqual(0, cv.countNonZero(b) + cv.countNonZero(g) + cv.countNonZero(r),
                             f"Images are not equal {image[0]}. Channels difference doesn't add up to 0")
    if __name__ == "__main__":
        unittest.main()
