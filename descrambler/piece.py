import cv2 as cv
import numpy as np
from descrambler.edge import Edge


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

    # main variables
    whole_piece = np.ndarray
    index = int

    # piece edits (used in the edge detection fitness algorithm)
    gray_piece = np.ndarray
    blur_piece = np.ndarray

    # original piece's edges (used in the color fitness algorithm)
    top_edge = np.ndarray
    right_edge = np.ndarray
    left_edge = np.ndarray
    bottom_edge = np.ndarray

    # fitness algorithm results
    top_edge_candidates = []
    bottom_edge_candidates = []
    right_edge_candidates = []
    left_edge_candidates = []

    top_right_corner = None
    top_left_corner = None
    bottom_right_corner = None
    bottom_left_corner = None


    def __init__(self, index, piece):
        self.whole_piece = piece
        self.index = index
        self.gray_piece = cv.cvtColor(piece, cv.COLOR_BGR2GRAY)
        self.blur_piece = cv.GaussianBlur(self.gray_piece, (5, 5), 0)  # image blur
        rows, cols = self.gray_piece.shape

        self.top_edge = self.whole_piece[0]
        self.bottom_edge = self.whole_piece[rows - 1]
        self.left_edge = self.whole_piece[:, 0]
        self.right_edge = self.whole_piece[:, cols - 1]

    def get_sorted_candidates(self, edge: Edge):
        if edge == Edge.TOP:
            return sorted(self.top_edge_candidates, key=lambda x: x.fitness_value)
        if edge == Edge.BOTTOM:
            return sorted(self.bottom_edge_candidates, key=lambda x: x.fitness_value)
        if edge == Edge.RIGHT:
            return sorted(self.right_edge_candidates, key=lambda x: x.fitness_value)
        if edge == Edge.LEFT:
            return sorted(self.left_edge_candidates, key=lambda x: x.fitness_value)
