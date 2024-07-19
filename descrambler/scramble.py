import os

import cv2 as cv
import numpy as np

from debug.debug import Debug as Debug
from debug.logger import Logger
from descrambler.candidate import Candidate
from descrambler.fitness_algorithm import ColorComparison
from descrambler.helper import Helper
from descrambler.piece import Piece
from descrambler.edge import Edge


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
    counter = 0

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

        if not os.path.exists(
                "C:\\Users\\risko\\Documents\\Coding\\Python\\myProjects\\PuzzleMaker\\images\\pieces\\single"):
            Logger.info("\nPieces folder doesn't exist")
            Logger.info("MAKING PIECES FOLDER...")
            os.makedirs("C:\\Users\\risko\\Documents\\Coding\\Python\\myProjects\\PuzzleMaker\\images\\pieces\\single")
            Logger.info("FOLDER 'pieces/single' MADE")

            Logger.info("\nPUTTING PIECES INTO FOLDER")
            for piece in self.image_pieces:
                Helper.get_certain_loading(current_progress=piece.index + 1, final_num=len(self.image_pieces),
                                           description="Loading pieces")
                cv.imwrite(
                    f'C:\\Users\\risko\\Documents\\Coding\\Python\\myProjects\\PuzzleMaker\\images\\pieces\\single\\{self.img_name}_{piece.index}.png',
                    piece.whole_piece)

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

    def make_image(self, image_arr: list[list[Piece]]):
        all_rows = []
        for i in range(self.rows):
            row_images = []
            for x in range(self.cols):
                row_images.append(image_arr[i][x].whole_piece)

            row = np.concatenate(row_images, axis=1)
            all_rows.append(row)

        grid = np.concatenate(all_rows)
        return grid

    def solve(self):
        Logger.info("\n-----SOLVING ALGORITHM-----")
        self.helper.add_timestamp()

        # compare the edge of each piece to each other to see which ones match the most
        Logger.info("-----GETTING LIKELIHOOD OF PIECES FITTING-----")
        self.get_piece_fitness()
        Logger.info("\nFITNESS VALUES CALCULATED")
        self.helper.add_timestamp()
        Logger.info(f'CALCULATING EDGE FITNESS VALUES: {self.helper.get_latest_timestamp_difference()} ms')
        Logger.info(f'Solving the image now')
        result = self.solve_image(0, [], [])
        print('results')
        index_candidates = list(map(lambda x: x.piece.index), result)
        print(index_candidates)

        if len(result) > 0:
            # create the rows first
            rows = []
            for i in range(self.rows):
                rows.append(result[i * self.cols: (i + 1) * self.cols])
        piece_arr = []
        for row in rows:
            piece_arr.append(list(map(lambda x: x.piece, row)))

        final_img = self.make_image(piece_arr)
        cv.imshow("final image", final_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite(
            "C:\\Users\\risko\\Documents\\Coding\\Python\\myProjects\\PuzzleMaker\\images\\unscrambledImages\\finished.png",
            final_img)

    @staticmethod
    def make_edges(piece: Piece):
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

    def sort_edges(self):
        for piece in self.image_pieces:
            # sort the pieces first
            piece.left_edge_candidates.sort(key=lambda a: a[1])
            piece.right_edge_candidates.sort(key=lambda a: a[1])
            piece.bottom_edge_candidates.sort(key=lambda a: a[1])
            piece.top_edge_candidates.sort(key=lambda a: a[1])

    def get_piece_fitness(self):  # checks how likely each piece is to be next to each other
        # fitness_algorithm = EdgeDetection(self.image_pieces, self.contrast)
        fitness_algorithm = ColorComparison(self.image_pieces)
        fitness_algorithm.get_piece_fitness()

    def get_possible_edge_indices(self, col: int) -> list[int]:
        possible_indices = []
        if col + 1 % self.cols != 0:
            possible_indices.append(col + 1)

        if col > self.cols - 1:
            possible_indices.append(col - self.cols)
        return possible_indices

    def solve_image(self, col: int, state: list[Candidate], used_pieces: list[int]) -> list:

        threshold = 10
        print(f'Used pieces: {used_pieces}')
        self.counter = self.counter + 1

        # finishing condition
        if col > self.cols * self.rows - 1:
            print(f'Went over {self.counter} pieces')
            return state

        if col == 0:
            # randomly choose the first piece from all the pieces
            for piece in self.image_pieces:
                # first convert the piece to a candidate to be able to use it in the algorithm
                piece_to_candidate = Candidate(piece, fitness_value=0)
                result = self.solve_image(col=col + 1, state=[piece_to_candidate],
                                          used_pieces=[piece_to_candidate.piece.index])
                if len(result) > 0:
                    return result

            return []
        else:
            if col % self.cols == 0:
                # the piece is in the beginning of the row and therefore there is no piece before it in the row
                # take the first piece of the row before and go through the bottom candidates
                # the piece from which the candidates will be taken for the next piece
                candidate_piece = state[col - self.cols]
                sorted_candidates = candidate_piece.piece.get_sorted_candidates(Edge.BOTTOM)
                possible_candidates = list(filter(lambda x: x.fitness_value < threshold, sorted_candidates))
                for candidate in possible_candidates:
                    if candidate.piece.index not in set(used_pieces):
                        new_state = state + [candidate]
                        new_used = used_pieces + [candidate.piece.index]
                        result = self.solve_image(col=col + 1, state=new_state, used_pieces=new_used)
                        if len(result) > 0:
                            return result
                return []
            else:
                # the piece is somewhere in the image where it always has a piece before it
                # the piece from which the candidates will be taken for the next piece
                candidate_piece = state[col - 1]
                sorted_candidates = candidate_piece.piece.get_sorted_candidates(Edge.RIGHT)
                possible_candidates = list(filter(lambda x: x.fitness_value < threshold, sorted_candidates))
                for candidate in possible_candidates:
                    if candidate.piece.index not in set(used_pieces):
                        # check if the piece is in a different row than the first one, if yes, compare more edges
                        passes_top_edge = True
                        if col > self.cols:
                            top_edge = state[col - self.cols]
                            for edge_candidate in top_edge.piece.get_sorted_candidates(Edge.BOTTOM):
                                if edge_candidate.fitness_value > threshold:
                                    passes_top_edge = False
                                    break
                                if edge_candidate.piece.index == candidate.piece.index:
                                    break
                                passes_top_edge = False

                        if passes_top_edge:
                            new_state = state + [candidate]
                            new_used = used_pieces + [candidate.piece.index]
                            result = self.solve_image(col=col + 1, state=new_state, used_pieces=new_used)
                            if len(result) > 0:
                                return result
                return []
