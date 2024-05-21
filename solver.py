import time

import cv2 as cv
import numpy as np
import os


class Helper:

    @staticmethod
    def current_milli_time():
        return round(time.time() * 1000)


class Piece:
    """
    - init:
        - params:
            - the position in the original picture
            - the value of the piece at the index position

        - variables:
            - whole_piece
            - index of the piece
            - top, right, bottom and left edges
            - average value of the whole piece

    """

    whole_piece = np.ndarray
    index = -1
    gray_piece = np.ndarray
    top_edge_candidates = []
    bottom_edge_candidates = []
    right_edge_candidates = []
    left_edge_candidates = []

    top_edge_avg = float
    bottom_edge_avg = float
    right_edge_avg = float
    left_edge_avg = float

    top_right_candidate = None
    top_left_candidate = None
    bottom_right_candidate = None
    bottom_left_candidate = None

    def __init__(self, index, piece):
        self.whole_piece = piece
        self.index = index
        self.gray_piece = cv.cvtColor(piece, cv.COLOR_BGR2GRAY)
        self.make_edges()

    def make_edges(self):
        top_edge = self.gray_piece[0]
        bottom_edge = self.gray_piece[self.gray_piece.shape[0] - 1]
        right_edge = self.gray_piece[:, self.gray_piece[0].shape[0] - 1]
        left_edge = self.gray_piece[:, 0]

        self.top_edge_avg = np.sum(top_edge) / top_edge.size
        self.bottom_edge_avg = np.sum(bottom_edge) / bottom_edge.size
        self.right_edge_avg = np.sum(right_edge) / right_edge.size
        self.left_edge_avg = np.sum(left_edge) / left_edge.size


class Scramble:
    """
    - init:
        - params:
            - path - path to the image
            - rows - number of rows in the image
            - cols - number of columns in the image


        - variables:
            - whole_image - stores the whole image loaded
            - rows - stores the number of rows
            - cols - stores the number of columns
            - image_pieces - stores all the pieces made from image


    """

    whole_image = None
    img_name = None
    rows = 1
    cols = 1
    image_pieces = []

    def __init__(self, path, rows, cols):
        self.img_name = os.path.basename(path)
        image = cv.imread(path, cv.IMREAD_COLOR)
        self.whole_image = image
        self.rows = rows
        self.cols = cols
        self.image_pieces = self.make_pieces()  # works

    def make_pieces(self) -> list[Piece]:  # gets the pieces from the scramble
        img_width = self.whole_image.shape[1]
        img_height = self.whole_image.shape[0]
        row_size = img_height // self.rows
        col_size = img_width // self.cols
        if img_height % self.rows != 0 or img_width % self.cols != 0:
            print(
                f'{self.img_name} isn\'t divisible by cols or rows. This will lead to some edges of the image to be left out!')
        index = 0
        pieces = []
        for i in range(self.rows):
            for j in range(self.cols):
                pieces.append(Piece(index, self.whole_image[i * row_size:(i + 1) * row_size, j * col_size:(j + 1) * col_size]))
                index = index + 1
        return pieces

    def solve(self):
        # combine all the pieces into two pieces
        # calculate which one fits the most
        self.get_pieces_fit_likelihood()
        print("---------PIECES FIT LIKELIHOOD---------")
        print(Helper.current_milli_time())

        self.get_corner_pieces()
        print("-----------CORNER PIECES-----------")
        print(Helper.current_milli_time())

        for piece in self.image_pieces:
            print(f"----------------Piece: {piece.index}----------------")
            print(
                f"Top: {piece.top_edge_candidates[0][0].index}\nBottom: {piece.bottom_edge_candidates[0][0].index}\nRight: {piece.right_edge_candidates[0][0].index}\nLeft: {piece.left_edge_candidates[0][0].index}")
            print(f"Top left: {piece.top_left_candidate.index}\nTop right: {piece.top_right_candidate.index}\nBottom right: {piece.bottom_right_candidate.index}\nBottom left: {piece.bottom_left_candidate.index}")
    def get_corner_pieces(self):
        # for every piece in the array, compare the pieces that most fit the top left, right and bottom left, right
        for piece in self.image_pieces:
            piece.top_left_candidate = self.get_highest_ranking_piece(piece.top_edge_candidates[0][0].left_edge_candidates, piece.left_edge_candidates[0][0].top_edge_candidates)
            piece.top_right_candidate = self.get_highest_ranking_piece(piece.top_edge_candidates[0][0].right_edge_candidates, piece.right_edge_candidates[0][0].top_edge_candidates)
            piece.bottom_right_candidate = self.get_highest_ranking_piece(piece.bottom_edge_candidates[0][0].right_edge_candidates, piece.right_edge_candidates[0][0].bottom_edge_candidates)
            piece.bottom_left_candidate = self.get_highest_ranking_piece(piece.bottom_edge_candidates[0][0].left_edge_candidates, piece.left_edge_candidates[0][0].bottom_edge_candidates)

    def get_highest_ranking_piece(self, first_arr: list[tuple[Piece, float]], second_arr: list[tuple[Piece, float]]) -> Piece:
        top_result = (self.image_pieces[0], 999999)
        for f_element in first_arr:
            piece = f_element
            for s_element in second_arr:
                if s_element[0].index is piece[0].index:
                    fit_likelihood = s_element[1] + piece[1]
                    if top_result[1] > fit_likelihood:
                        top_result = (piece[0], fit_likelihood)
        return top_result[0]

    def get_pieces_fit_likelihood(self):  # checks how likely each piece is to be next to each other
        color_diff_thresh = 50

        for piece_one in self.image_pieces:
            right_edge_fit_likelihood = []
            left_edge_fit_likelihood = []
            top_edge_fit_likelihood = []
            bottom_edge_fit_likelihood = []
            for piece_two in self.image_pieces:
                if piece_two != piece_one:  # check whether the pieces aren't the same piece
                    horizontal_edge_fit = np.concatenate(
                        [piece_two.gray_piece, piece_one.gray_piece, piece_two.gray_piece], axis=1)
                    horizontal_edge_likelihood = self.get_piece_edge_likelihood(horizontal_edge_fit, is_horizontal=True)

                    if abs(piece_one.left_edge_avg - piece_two.right_edge_avg) < color_diff_thresh:
                        left_edge_fit_likelihood.append((piece_two, horizontal_edge_likelihood[0]))

                    if abs(piece_one.right_edge_avg - piece_two.left_edge_avg) < color_diff_thresh:
                        right_edge_fit_likelihood.append((piece_two, horizontal_edge_likelihood[1]))

                    vertical_edge_fit = np.concatenate(
                        [piece_two.gray_piece, piece_one.gray_piece, piece_two.gray_piece])
                    vertical_edge_likelihood = self.get_piece_edge_likelihood(vertical_edge_fit, is_horizontal=False)

                    if abs(piece_one.top_edge_avg - piece_two.bottom_edge_avg) < color_diff_thresh:
                        top_edge_fit_likelihood.append((piece_two, vertical_edge_likelihood[0]))

                    if abs(piece_one.bottom_edge_avg - piece_two.top_edge_avg) < color_diff_thresh:
                        bottom_edge_fit_likelihood.append((piece_two, vertical_edge_likelihood[1]))

            left_edge_fit_likelihood.sort(key=lambda a: a[1])
            right_edge_fit_likelihood.sort(key=lambda a: a[1])
            bottom_edge_fit_likelihood.sort(key=lambda a: a[1])
            top_edge_fit_likelihood.sort(key=lambda a: a[1])

            piece_one.left_edge_candidates = left_edge_fit_likelihood
            piece_one.right_edge_candidates = right_edge_fit_likelihood
            piece_one.bottom_edge_candidates = bottom_edge_fit_likelihood
            piece_one.top_edge_candidates = top_edge_fit_likelihood

    def get_piece_edge_likelihood(self, image: np.ndarray, is_horizontal: bool) -> tuple[float, float]:
        image_blur = cv.GaussianBlur(image, (3, 3), 0)  # image blur
        edges = cv.Canny(image=image_blur, threshold1=30, threshold2=75)  # Canny Edge Detection

        if is_horizontal:
            edges_width = edges.shape[1]
            left_edge = np.array(edges[:, (edges_width // 3) - 1: (edges_width // 3) + 1])
            right_edge = np.array(edges[:, (edges_width // 3) * 2 - 1: (edges_width // 3) * 2 + 1])

            left_edge_length = left_edge.size
            right_edge_length = right_edge.size

            left_edge_likelihood = np.sum(left_edge) / left_edge_length
            right_edge_likelihood = np.sum(right_edge) / right_edge_length

            return left_edge_likelihood, right_edge_likelihood
        else:
            edges_height = edges.shape[0]
            top_edge = np.array(edges[(edges_height // 3) - 1:(edges_height // 3) + 1])
            bottom_edge = np.array(edges[(edges_height // 3) * 2 - 1:(edges_height // 3) * 2 + 1])

            top_edge_length = top_edge.size
            bottom_edge_length = bottom_edge.size

            top_edge_likelihood = np.sum(top_edge) / top_edge_length
            bottom_edge_likelihood = np.sum(bottom_edge) / bottom_edge_length

            return top_edge_likelihood, bottom_edge_likelihood


IMAGE = 'black_white_test.png'
image_path = "scrambledImages/" + IMAGE
algorithm = Scramble(image_path, rows=9, cols=8)
print(Helper.current_milli_time())
algorithm.solve()
