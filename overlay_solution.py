import cv2 as cv
import sys
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
    top_edge = []
    bottom_edge = []
    right_edge = []
    left_edge = []
    overall_value = 0
    whole_piece = [0, 0, 0]
    index = -1

    def __init__(self, index, piece):
        self.whole_piece = piece
        self.index = index
        self.top_edge = piece[0]
        self.bottom_edge = piece[len(piece) - 1]
        self.right_edge = piece[:, len(piece[0]) - 1]
        self.left_edge = piece[:, 0]
        self.get_overall_piece_value()

    def get_overall_piece_value(self):
        total_sum = np.sum(np.ravel(self.whole_piece))
        self.overall_value = total_sum / (self.whole_piece.shape[0] * self.whole_piece.shape[1])


class Algorithm:
    """
    - init
        - params
            - original image
            - number of rows
            - number of cols
    - functions
        - make quadrons
            - makes 5 piece quadrons based on the edges and the overall values of the pieces
        - quadron to image (for testing purposes)
            - just returns an image
        - overlay quadrons
            - overlays the quadrons on top of each other
            - the most overlayed quadrons are most likely the most accurate ones

    """

    image = []
    rows = -1
    cols = -1

    pieces = []
    img_height = -1
    img_width = -1
    row_size = -1
    col_size = -1

    def __init__(self, image, rows, cols):
        self.image = image
        self.rows = rows
        self.cols = cols
        self.img_width = image.shape[1]
        self.img_height = image.shape[0]
        self.row_size = self.img_height // self.rows
        self.col_size = self.img_width // self.cols

        self.pieces = self.make_pieces()

    def make_pieces(self):  # gets the pieces from the image
        index = 0
        pieces = []
        for i in range(self.rows):
            for j in range(self.cols):
                pieces.append(Piece(index, self.image[i * self.row_size:(i + 1) * self.row_size,
                                           j * self.col_size:(j + 1) * self.col_size]))
                index = index + 1
        return pieces

    def make_quadrons(self):  # function retuns done quadrons
        quadrons = self.get_quadrons()
        # quadron_images = []
        # for quadron in quadrons:
        #     quadron_images.append(self.quadron_to_image(quadron, quadrons[quadron]))

        # return quadron_images

    def quadron_to_image(self, main_piece: Piece,
                         quadron: dict) -> np.array:  # turns the pieces into images to be later viewed for testing purposes
        right_piece = quadron['right'][0]
        bottom_piece = quadron['bottom'][0]
        left_piece = quadron['left'][0]
        top_piece = quadron['top'][0]
        black_piece = np.zeros(main_piece.whole_piece.shape)

        arrangement = [[black_piece, top_piece.whole_piece, black_piece],
                       [left_piece.whole_piece, main_piece.whole_piece, right_piece.whole_piece],
                       [black_piece, bottom_piece.whole_piece, black_piece]]

        all_rows = []
        for i in range(len(arrangement)):
            row_pieces = arrangement[i]
            row = np.concatenate(row_pieces, axis=1)
            all_rows.append(row)

        final_img = np.concatenate(all_rows)
        return final_img

    def get_quadrons(self):
        edge_value_weight = 300
        overall_value_weight = 1
        quadrons = {}
        i = 0
        for piece in self.pieces:
            quadrons[piece] = {'right': [], 'bottom': [], 'top': [], 'left': []}
            quadrons[piece]['right'] = sorted(self.pieces,
                                              key=lambda x: edge_value_weight * self.compare_edges(
                                                  piece.right_edge,
                                                  x.left_edge) + overall_value_weight * self.compare_overall_values(
                                                  piece.overall_value, x.overall_value))
            quadrons[piece]['right'].remove(piece)
            quadrons[piece]['bottom'] = sorted(self.pieces,
                                               key=lambda x: edge_value_weight * self.compare_edges(
                                                   piece.bottom_edge,
                                                   x.top_edge) + overall_value_weight * self.compare_overall_values(
                                                   piece.overall_value, x.overall_value))
            quadrons[piece]['bottom'].remove(piece)
            quadrons[piece]['top'] = sorted(self.pieces,
                                            key=lambda x: edge_value_weight * self.compare_edges(
                                                piece.top_edge,
                                                x.bottom_edge) + overall_value_weight * self.compare_overall_values(
                                                piece.overall_value, x.overall_value))
            quadrons[piece]['top'].remove(piece)
            quadrons[piece]['left'] = sorted(self.pieces,
                                             key=lambda x: edge_value_weight * self.compare_edges(
                                                 piece.left_edge,
                                                 x.right_edge) + overall_value_weight * self.compare_overall_values(
                                                 piece.overall_value, x.overall_value))
            quadrons[piece]['left'].remove(piece)
            i = i + 1
            self.update_loading_bar(i, len(self.pieces), "Getting Quadrons")

        return quadrons

    def overlay_solutions(self, quadrons):
        pass

    def update_loading_bar(self, current_progress: int, final_num: int, description: str):
        t = "█"
        a = "."
        loading_position = ''
        if current_progress % 5 == 0:
            loading_position = '▁▃▅▆█'
        elif current_progress %5 == 1:
            loading_position = '█▁▃▅▆'
        elif current_progress %5 == 2:
            loading_position = '▆█▁▃▅'
        elif current_progress %5 == 3:
            loading_position = '▅▆█▁▃'
        elif current_progress %5 == 4:
            loading_position = '▃▅▆█▁'

        sys.stdout.write(
            f"\r{description}[{t * current_progress}{a * (final_num - current_progress)}] | {loading_position} {current_progress}/{final_num}")
        sys.stdout.flush()



    def compare_edges(self, edge_one, edge_two):
        total_match = 0
        tolerance = 10
        for i in range(len(edge_one)):
            edge_one_pixel = edge_one[i]
            edge_two_pixel = edge_two[i]
            match = abs(edge_one_pixel - edge_two_pixel)
            current_match = tolerance + 1

            if match == 0:
                current_match = 0
            elif match <= tolerance:
                current_match = match

            total_match = total_match + current_match

        return total_match

    def compare_overall_values(self, value_one, value_two):
        return abs(value_one - value_two)


IMAGE = 'black_white_test.png'
img = cv.imread("scrambledImages/" + IMAGE, cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

algorithm = Algorithm(img, rows=9, cols=8)
test_images = algorithm.make_quadrons()
i = 0
for image in test_images:
    cv.imwrite("triplets/black_white_test/" + str(i) + ".jpg", image)
    i = i + 1
