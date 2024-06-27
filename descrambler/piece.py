import cv2 as cv
import numpy as np


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
    index = int
    gray_piece = np.ndarray
    blur_piece = np.ndarray
    top_edge_candidates = []
    bottom_edge_candidates = []
    right_edge_candidates = []
    left_edge_candidates = []

    top_right_candidates = []
    top_left_candidates = []
    bottom_right_candidates = []
    bottom_left_candidates = []

    def __init__(self, index, piece):
        self.whole_piece = piece
        self.index = index
        self.gray_piece = cv.cvtColor(piece, cv.COLOR_BGR2GRAY)
        self.blur_piece = cv.GaussianBlur(self.gray_piece, (5, 5), 0)  # image blur
