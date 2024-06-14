import cv2 as cv
import numpy as np
import os
from descrambler.helper import Helper
from debug.debug import Debug as Debug
from debug.logger import Logger


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
    contrast = None
    helper = Helper()

    def __init__(self, path, rows, cols):
        self.helper.add_timestamp()

        self.img_name = os.path.basename(path)
        self.img_name = self.img_name[:len(self.img_name) - 4]
        image = cv.imread(path, cv.IMREAD_COLOR)
        self.whole_image = image
        self.rows = rows
        self.cols = cols
        self.image_pieces = self.make_pieces()

        img_grey = cv.cvtColor(self.whole_image, cv.COLOR_BGR2GRAY)
        self.contrast = img_grey.std()

        Logger.info(f"Contrast of image: {self.contrast}")

        if not os.path.exists("../images/pieces/single"):
            Logger.info("\nPieces folder doesn't exist")
            Logger.info("MAKING PIECES FOLDER...")
            os.makedirs("../images/pieces/single")
            Logger.info("FOLDER 'pieces/single' MADE")

            Logger.info("\nPUTTING PIECES INTO FOLDER")
            for piece in self.image_pieces:
                Helper.get_certain_loading(current_progress=piece.index + 1, final_num=len(self.image_pieces),
                                           description="Loading pieces")
                cv.imwrite(f'pieces/single/{self.img_name}_{piece.index}.png', piece.whole_piece)

            Logger.info("\nPIECES PUT INTO FOLDER 'pieces/single'")
        else:
            Logger.info("Pieces folder exists. Remove to update for latest pieces...")

        self.helper.add_timestamp()
        Logger.info(f'SETTING PIECES UP: {self.helper.get_latest_timestamp_difference()} ms')

    def get_image(self):
        all_rows = []
        for i in range(self.rows):
            row_images = []
            for x in range(self.cols):
                row_images.append(self.image_pieces[i * self.cols + x].whole_piece)

            row = np.concatenate(row_images, axis=1)
            all_rows.append(row)

        grid = np.concatenate(all_rows)
        return grid

    def make_pieces(self) -> list[Piece]:  # gets the pieces from the scramble based on the given cols and rows
        Logger.info("-----LOADING IMAGE PIECES BASED ON ARGUMENTS-----")
        img_width = self.whole_image.shape[1]
        img_height = self.whole_image.shape[0]
        row_size = img_height // self.rows
        col_size = img_width // self.cols

        Logger.info(f'IMAGE HEIGHT: {img_height}, IMAGE WIDTH: {img_width}')
        Logger.info(f'ROWS: {self.rows}, COLS: {self.cols}')
        Logger.info(f'ROW SIZE: {row_size}, COL SIZE: {col_size}')
        if img_height % self.rows != 0 or img_width % self.cols != 0:
            Logger.info(
                f'{self.img_name} isn\'t divisible by cols or rows. This will lead to some edges of the image to be left out!')
            Logger.info(f'Rows * row_size: {row_size * self.rows}')
            Logger.info(f'Cols * col_size {col_size * self.cols}')

        index = 0
        pieces = []
        for i in range(self.rows):
            for j in range(self.cols):
                pieces.append(  # Creating of piece objects
                    Piece(index, self.whole_image[i * row_size:(i + 1) * row_size, j * col_size:(j + 1) * col_size]))
                index = index + 1
        return pieces

    def solve(self):
        Logger.info("\n-----SOLVING ALGORITHM-----")
        self.helper.add_timestamp()

        # compare the edge of each piece to each other to see which ones match the most
        Logger.info("-----GETTING LIKELIHOOD OF PIECES FITTING-----")
        self.get_piece_fitness()
        Logger.info("\nFITNESS VALUES CALCULATED")
        self.helper.add_timestamp()
        Logger.info(f'CALCULATING EDGE FITNESS VALUES: {self.helper.get_latest_timestamp_difference()} ms')

        # get the corners based on the made pieces already
        # self.get_corners_fitness()

        # review the edge pieces using the corner pieces
        # self.review_edge_fitness()

        if Debug.DEBUG:

            total_fitness = 0
            for piece in self.image_pieces:
                # sort the pieces first
                piece.left_edge_candidates.sort(key=lambda a: a[1])
                piece.right_edge_candidates.sort(key=lambda a: a[1])
                piece.bottom_edge_candidates.sort(key=lambda a: a[1])
                piece.top_edge_candidates.sort(key=lambda a: a[1])

                Logger.info(f"----------------Piece {piece.index}----------------")
                Logger.info(f"Top: {piece.top_edge_candidates[0][0].index}----------{piece.top_edge_candidates[0][1]}")

                Logger.info(
                    f"Bottom: {piece.bottom_edge_candidates[0][0].index}----------{piece.bottom_edge_candidates[0][1]}")

                Logger.info(
                    f"Right: {piece.right_edge_candidates[0][0].index}----------{piece.right_edge_candidates[0][1]}")

                Logger.info(
                    f"Left: {piece.left_edge_candidates[0][0].index}----------{piece.left_edge_candidates[0][1]}")

                total_fitness = total_fitness + piece.top_edge_candidates[0][1] + piece.bottom_edge_candidates[0][1] + \
                                piece.right_edge_candidates[0][1] + piece.left_edge_candidates[0][1]
            Logger.info("------------------------------------------------------")
            Logger.info(f"Overall fitness: {total_fitness / (self.rows * self.cols * 4)}")

        if not os.path.exists("../images/pieces/edges"):
            Logger.info("\nEdge pieces folder doesn't exist")
            Logger.info("MAKING EDGE PIECES FOLDER...")
            os.makedirs("../images/pieces/edges")
            Logger.info("FOLDER 'pieces/edges' MADE")

            Logger.info("\nPUTTING PIECES INTO FOLDER")
            for piece in self.image_pieces:
                Helper.get_certain_loading(current_progress=piece.index + 1, final_num=len(self.image_pieces),
                                           description="Loading pieces")
                edges = self.make_edges(piece)
                cv.imwrite(f'pieces/edges/{self.img_name}_{piece.index}.png', edges)

            Logger.info("\nPIECES PUT INTO FOLDER 'pieces/single'")
        else:
            Logger.info("Pieces folder exists. Remove to update for latest pieces...")

    def make_edges(self, piece: Piece):
        dark_piece = np.zeros(piece.whole_piece.shape, dtype=int)

        all_rows = [[dark_piece, piece.top_edge_candidates[0][0].whole_piece, dark_piece],
                    [piece.left_edge_candidates[0][0].whole_piece, piece.whole_piece,
                     piece.right_edge_candidates[0][0].whole_piece],
                    [dark_piece, piece.bottom_edge_candidates[0][0].whole_piece, dark_piece]]
        row_images = []
        for row in all_rows:
            row_image = np.concatenate(row, axis=1)
            row_images.append(row_image)

        grid = np.concatenate(row_images)
        return grid

    def overlay_pieces(self):
        for piece in self.image_pieces:
            Logger.info()

    def get_piece_fitness(self):  # checks how likely each piece is to be next to each other

        for piece_one in self.image_pieces:
            Helper.get_certain_loading(current_progress=piece_one.index + 1, final_num=len(self.image_pieces),
                                       description="Getting fitness values of piece edges: ")
            right_edge_fit_likelihood = []
            left_edge_fit_likelihood = []
            top_edge_fit_likelihood = []
            bottom_edge_fit_likelihood = []
            for piece_two in self.image_pieces:
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
        # image_blur = cv.GaussianBlur(image, (5, 5), 0)  # image blur
        edges = cv.Canny(image=image, threshold1=30, threshold2=self.contrast)  # Canny Edge Detection

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
