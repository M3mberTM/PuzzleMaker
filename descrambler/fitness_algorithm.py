import cv2 as cv
import numpy as np

from descrambler.piece import Piece
from descrambler.helper import Helper


class EdgeDetection:
    pieces = list
    contrast = int

    def __init__(self, pieces: list[Piece], contrast):
        self.pieces = pieces
        self.contrast = contrast

    def get_piece_fitness(self):  # checks how likely each piece is to be next to each other

        for piece_one in self.pieces:
            Helper.get_certain_loading(current_progress=piece_one.index + 1, final_num=len(self.pieces),
                                       description="Getting fitness values of piece edges: ")
            right_edge_fit_likelihood = []
            left_edge_fit_likelihood = []
            top_edge_fit_likelihood = []
            bottom_edge_fit_likelihood = []
            for piece_two in self.pieces:
                if piece_two != piece_one:  # check whether the pieces aren't the same piece

                    # concatenating horizontally on both sides at once to save on computing power
                    horizontal_edge_fit = np.concatenate(
                        [piece_two.gray_piece, piece_one.gray_piece, piece_two.gray_piece], axis=1)
                    horizontal_edge_likelihood = self.get_edge_fitness(horizontal_edge_fit, is_horizontal=True)

                    left_edge_fit_likelihood.append((piece_two, horizontal_edge_likelihood[0]))

                    right_edge_fit_likelihood.append((piece_two, horizontal_edge_likelihood[1]))

                    vertical_edge_fit = np.concatenate(
                        [piece_two.gray_piece, piece_one.gray_piece, piece_two.gray_piece])
                    vertical_edge_likelihood = self.get_edge_fitness(vertical_edge_fit, is_horizontal=False)

                    top_edge_fit_likelihood.append((piece_two, vertical_edge_likelihood[0]))

                    bottom_edge_fit_likelihood.append((piece_two, vertical_edge_likelihood[1]))

            piece_one.left_edge_candidates = left_edge_fit_likelihood
            piece_one.right_edge_candidates = right_edge_fit_likelihood
            piece_one.bottom_edge_candidates = bottom_edge_fit_likelihood
            piece_one.top_edge_candidates = top_edge_fit_likelihood

    def get_edge_fitness(self, image: np.ndarray, is_horizontal: bool) -> tuple[float, float]:
        image_blur = cv.GaussianBlur(image, (5, 5), 0)  # image blur
        edges = cv.Canny(image=image_blur, threshold1=30, threshold2=70 - self.contrast)  # Canny Edge Detection

        if is_horizontal:
            edges_width = edges.shape[1]
            left_edge = np.array(edges[:, (edges_width // 3) - 1: (edges_width // 3)])
            right_edge = np.array(edges[:, (edges_width // 3) * 2 - 1: (edges_width // 3) * 2])

            left_edge_length = left_edge.size
            right_edge_length = right_edge.size

            left_edge_likelihood = np.sum(left_edge) / left_edge_length
            right_edge_likelihood = np.sum(right_edge) / right_edge_length

            return left_edge_likelihood, right_edge_likelihood
        else:
            edges_height = edges.shape[0]
            top_edge = np.array(edges[(edges_height // 3) - 1:(edges_height // 3)])
            bottom_edge = np.array(edges[(edges_height // 3) * 2 - 1:(edges_height // 3) * 2])

            top_edge_length = top_edge.size
            bottom_edge_length = bottom_edge.size

            top_edge_likelihood = np.sum(top_edge) / top_edge_length
            bottom_edge_likelihood = np.sum(bottom_edge) / bottom_edge_length

            return top_edge_likelihood, bottom_edge_likelihood


class ColorComparison:
    pieces = list[Piece]

    def __init__(self, pieces: list[Piece]):
        self.pieces = pieces

    def get_piece_fitness(self):

        for piece_one in self.pieces:
            Helper.get_certain_loading(current_progress=piece_one.index + 1, final_num=len(self.pieces),
                                       description="Getting fitness values of piece edges: ")
            right_edge_fit_likelihood = []
            left_edge_fit_likelihood = []
            top_edge_fit_likelihood = []
            bottom_edge_fit_likelihood = []

            for piece_two in self.pieces:
                if piece_one.index != piece_two.index:  # check whether pieces aren't the same

                    right_edge_fitness = self.compare_edges(piece_one.right_edge, piece_two.left_edge)
                    left_edge_fitness = self.compare_edges(piece_one.left_edge, piece_two.right_edge)
                    top_edge_fitness = self.compare_edges(piece_one.top_edge, piece_two.bottom_edge)
                    bottom_edge_fitness = self.compare_edges(piece_one.bottom_edge, piece_two.top_edge)

                    right_edge_fit_likelihood.append((piece_two, right_edge_fitness))
                    left_edge_fit_likelihood.append((piece_two, left_edge_fitness))
                    top_edge_fit_likelihood.append((piece_two, top_edge_fitness))
                    bottom_edge_fit_likelihood.append((piece_two, bottom_edge_fitness))

            piece_one.left_edge_candidates = left_edge_fit_likelihood
            piece_one.right_edge_candidates = right_edge_fit_likelihood
            piece_one.bottom_edge_candidates = bottom_edge_fit_likelihood
            piece_one.top_edge_candidates = top_edge_fit_likelihood

    @staticmethod
    def compare_edges(first_edge: np.ndarray, second_edge: np.ndarray):
        first_edge = first_edge.astype('int')
        second_edge = second_edge.astype('int')
        difference = np.subtract(first_edge, second_edge)
        squared = np.power(difference / 255.0, 2)
        color_difference_per_row = np.sum(squared, axis=1)
        total_difference = np.sum(color_difference_per_row, axis=0)

        value = np.sqrt(total_difference)
        return value
