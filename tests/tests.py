import unittest
from descrambler.scramble import Scramble as Scramble
from os import listdir
from os.path import isfile, join
import cv2 as cv
import numpy as np
from descrambler.fitness_algorithm import ColorComparison

class TestDescrambling(unittest.TestCase):
    dir_files = list
    images = list

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

    def test_fitness_func(self):

        for image in self.images:

            # make scramble object
            scramble = Scramble(image[0], image[1][0], image[1][1])
            scramble.get_piece_fitness()
            scramble.sort_edges()
            strictness = (scramble.rows * scramble.cols) // 2
            results = []

            for piece in scramble.image_pieces:
                # get the top 5 from each edge
                index = piece.index

                top_edges = piece.top_edge_candidates[0:strictness]
                top_edges = list(map(lambda p: p[0].index, top_edges))
                bottom_edges = piece.bottom_edge_candidates[0:strictness]
                bottom_edges = list(map(lambda p: p[0].index, bottom_edges))
                right_edges = piece.right_edge_candidates[0:strictness]
                right_edges = list(map(lambda p: p[0].index, right_edges))
                left_edges = piece.left_edge_candidates[0:strictness]
                left_edges = list(map(lambda p: p[0].index, left_edges))

                print(f'\n----- PIECE {index} -----')
                # compare left edge if the piece isn't at the left edge originally
                if index % scramble.cols != 0:
                    correct_val = index - 1
                    result = correct_val in left_edges
                    results.append(result)
                    if result:
                        print('Left - OK')
                    else:
                        print('Left - NOT OK')

                # compare right edge if the piece isn't at the right edge originally
                if (index + 1) % scramble.cols != 0:
                    correct_val = index + 1
                    result = correct_val in right_edges
                    results.append(result)
                    if result:
                        print('Right - OK')
                    else:
                        print('Right - NOT OK')

                # compare top edge if the piece isn't at the top edge originally
                if index >= scramble.cols:
                    correct_val = index - scramble.cols
                    result = correct_val in top_edges
                    results.append(result)
                    if result:
                        print('Top - OK')
                    else:
                        print('Top - NOT OK')

                # compare bottom edge if the piece isn't at the bottom edge originally
                if index < (scramble.cols * scramble.rows) - scramble.cols:
                    correct_val = index + scramble.cols
                    result = correct_val in bottom_edges
                    results.append(result)
                    if result:
                        print('Bottom - OK')
                    else:
                        print('Bottom - NOT OK')

            self.assertEqual((scramble.rows * scramble.cols * 4) - 2 * scramble.rows - 2 * scramble.cols, sum(results))
            print(f'Image {image} ---------- OK')

    def test_color_comparison(self):
        piece_one = cv.imread('../images/pieces/single/08_851131_851131_1_006_001_1.png', cv.IMREAD_COLOR)
        piece_two = cv.imread('../images/pieces/single/08_851131_851131_1_006_001_2.png', cv.IMREAD_COLOR)
        piece_three = cv.imread('../images/pieces/single/08_851131_851131_1_006_001_0.png', cv.IMREAD_COLOR)

        rows, cols, _ = piece_one.shape
        edge_one = piece_one[:, cols-1]
        edge_two = piece_two[:, 0]
        edge_three = piece_three[:, 0]

        comparison_one = ColorComparison.compare_edges(edge_one, edge_two)
        comparison_two = ColorComparison.compare_edges(edge_one, edge_three)

        print(comparison_one)
        print(comparison_two)
        self.assertTrue(comparison_one < comparison_two)

    def test_closest_fitting_index(self):

        worst_indices = []
        average_indices = []

        for image in self.images:

            # make scramble object
            scramble = Scramble(image[0], image[1][0], image[1][1])
            scramble.get_piece_fitness()
            scramble.sort_edges()
            highest_index = 0
            sum_of_indices = 0
            all_indices = []

            for piece in scramble.image_pieces:

                index = piece.index

                top_edges = piece.top_edge_candidates
                top_edges = list(map(lambda p: p[0].index, top_edges))
                bottom_edges = piece.bottom_edge_candidates
                bottom_edges = list(map(lambda p: p[0].index, bottom_edges))
                right_edges = piece.right_edge_candidates
                right_edges = list(map(lambda p: p[0].index, right_edges))
                left_edges = piece.left_edge_candidates
                left_edges = list(map(lambda p: p[0].index, left_edges))

                # compare left edge if the piece isn't at the left edge originally
                if index % scramble.cols != 0:
                    correct_val = index - 1
                    result = left_edges.index(correct_val)
                    all_indices.append(result)
                    if result > highest_index:
                        highest_index = result

                # compare right edge if the piece isn't at the right edge originally
                if (index + 1) % scramble.cols != 0:
                    correct_val = index + 1
                    result = right_edges.index(correct_val)
                    all_indices.append(result)
                    if result > highest_index:
                        highest_index = result

                # compare top edge if the piece isn't at the top edge originally
                if index >= scramble.cols:
                    correct_val = index - scramble.cols
                    result = top_edges.index(correct_val)
                    all_indices.append(result)
                    if result > highest_index:
                        highest_index = result

                # compare bottom edge if the piece isn't at the bottom edge originally
                if index < (scramble.cols * scramble.rows) - scramble.cols:
                    correct_val = index + scramble.cols
                    result = bottom_edges.index(correct_val)
                    all_indices.append(result)
                    if result > highest_index:
                        highest_index = result

            worst_indices.append(highest_index)  # append the highest index for an image
            average_indices.append(np.sum(all_indices) / len(all_indices))
        print(f'Worst: {worst_indices}')
        print(f'Average: {average_indices}')

    if __name__ == "__main__":
        unittest.main()
