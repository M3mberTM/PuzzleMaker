import math

import cv2 as cv
import numpy as np
from typing import Tuple
import random
import datetime


class Piece:
    """
    - init:
        - params:
            - the position in the original picture
            - the value of the piece at the index position

    edges are all in grayscale for easier and quicker comparison later on
    """
    top_edge = []
    top_edge_value = [0, 0, 0]
    bottom_edge = []
    bottom_edge_value = [0, 0, 0]
    right_edge = []
    right_edge_value = [0, 0, 0]
    left_edge = []
    left_edge_value = [0, 0, 0]
    overall_value = [0, 0, 0]
    whole_piece = [0, 0, 0]
    index = -1

    def __init__(self, index, piece):
        self.whole_piece = piece
        self.index = index
        self.top_edge = piece[0]
        self.bottom_edge = piece[len(piece) - 1]
        self.right_edge = piece[:, len(piece[0]) - 1]
        self.left_edge = piece[:, 0]
        self.get_edge_values()
        self.overall_value = self.get_overall_piece_value(piece)

    def get_edge_values(self):
        b, g, r = self.get_edge_bgr_vals(self.top_edge)
        self.top_edge_value = [b, g, r]
        b, g, r = self.get_edge_bgr_vals(self.bottom_edge)
        self.bottom_edge_value = [b, g, r]
        b, g, r = self.get_edge_bgr_vals(self.right_edge)
        self.right_edge_value = [b, g, r]
        b, g, r = self.get_edge_bgr_vals(self.left_edge)
        self.left_edge_value = [b, g, r]

    def get_edge_bgr_vals(self, edge):
        blue_val = 0
        green_val = 0
        red_val = 0
        for value in edge:
            blue_val = blue_val + value[0]
            green_val = green_val + value[1]
            red_val = red_val + value[2]

        return blue_val, green_val, red_val

    def get_overall_piece_value(self, piece):
        whole_value = [0, 0, 0]
        for row in piece:
            for value in row:
                whole_value[0] = whole_value[0] + value[0]
                whole_value[1] = whole_value[1] + value[1]
                whole_value[2] = whole_value[2] + value[2]
        whole_value[0] = whole_value[0] / len(piece)
        whole_value[1] = whole_value[1] / len(piece)
        whole_value[2] = whole_value[2] / len(piece)
        return whole_value


class Generation:
    """
    init
    parameters: top solutions from previous gen, how many solutions to return in fitness

    fitness function
        compare edges
            returns likelihood of the squares being close by checking how far away the colors are from each other
            always divide the result by the number of corners since somewhere we only look at one corner
        compare colors
            compares the average colors of the pieces next to each other to see if they match
        sort
        returns top x solutions in format (quality_value, solution)
    """
    solutions = []
    return_num = -1
    rows = -1
    cols = -1
    threshold = -1

    def __init__(self, solutions: list[list[Piece]], top_sol_num: int, rows: int, cols: int, threshold: int):
        self.solutions = solutions
        self.return_num = top_sol_num
        self.rows = rows
        self.cols = cols
        self.threshold = threshold

    def get_ranked_solutions(self) -> list:
        ranked_solutions = []
        for solution in self.solutions:
            ranked_solutions.append(self.fitness(solution))
        ranked_solutions.sort(key=lambda x: x[0])
        return ranked_solutions[:self.return_num]

    def fitness(self, solution: list[Piece]) -> Tuple[float, list]:  # TODO add more parameters by which to format pieces (average color value in the whole piece)
        edge_proximity_weight = 1
        edge_proximity_match = self.get_edge_proximity_match(solution) * edge_proximity_weight
        proximity_color_weight = 0.5
        proximity_color_match = self.color_comparison(solution) * proximity_color_weight
        return edge_proximity_match + proximity_color_match, solution

    def get_edge_proximity_match(self, solution) -> int:  # go through each piece and add up the compare edges values
        grid_likelihood = 0
        for index in range(len(solution)):  # pieces
            if (index + 1) % self.cols != 0:
                edge_likelihood = 0
                if index + 1 <= self.cols * (
                        self.rows - 1):  # if it isn't the last row, compare both the bottom and right edges
                    edge_likelihood = edge_likelihood + self.compare_edge_values(solution[index].bottom_edge_value,
                                                                                 solution[
                                                                                     index + self.cols].top_edge_value)
                    edge_likelihood = edge_likelihood + self.compare_overall_values(solution[index].overall_value,
                                                                                    solution[
                                                                                        index + self.cols].overall_value)
                    edge_likelihood = edge_likelihood + self.compare_edge_values(solution[index].right_edge_value,
                                                                                 solution[index + 1].left_edge_value)
                    edge_likelihood = edge_likelihood + self.compare_overall_values(solution[index].overall_value,
                                                                                    solution[index + 1].overall_value)

                    edge_likelihood = edge_likelihood / 2  # divide the edge likelihood by number of edges compared for more accurate results

                else:  # if the element is in last row, don't compare bottom edge

                    edge_likelihood = edge_likelihood + self.compare_edge_values(solution[index].right_edge_value,
                                                                                 solution[index + 1].left_edge_value)
                    edge_likelihood = edge_likelihood + self.compare_overall_values(solution[index].overall_value,
                                                                                    solution[index + 1].overall_value)
                grid_likelihood = grid_likelihood + edge_likelihood

            else:  # if it's the last element in row and is not the last element of the picture, compare only the down edge
                if index + 1 <= self.cols * (self.rows - 1):
                    edge_likelihood = self.compare_edge_values(solution[index].bottom_edge_value,
                                                               solution[index + self.cols].top_edge_value)
                    edge_likelihood = edge_likelihood + self.compare_overall_values(solution[index].overall_value,
                                                                                    solution[
                                                                                        index + self.cols].overall_value)
                    grid_likelihood = grid_likelihood + edge_likelihood

        return grid_likelihood

    def color_comparison(self, solution) -> int:  # go through each piece and add up the overall piece color
        color_compared = 0
        for index in range(len(solution)):  # pieces
            if (index + 1) % self.cols != 0:
                proximity_value = 0
                if index + 1 <= self.cols * (
                        self.rows - 1):  # if it isn't the last row, compare both the bottom and right pieces

                    proximity_value = proximity_value + self.compare_overall_values(solution[index].overall_value,
                                                                                    solution[
                                                                                        index + self.cols].overall_value)

                    proximity_value = proximity_value + self.compare_overall_values(solution[index].overall_value,
                                                                                    solution[index + 1].overall_value)

                    proximity_value = proximity_value / 2  # divide the final value by the number of operations for accurate results

                else:  # if the element is in last row, don't compare bottom piece as there is none

                    proximity_value = proximity_value + self.compare_overall_values(solution[index].overall_value,
                                                                                    solution[index + 1].overall_value)
                color_compared = color_compared + proximity_value

            else:  # if it's the last element in row and is not the last element of the picture, compare only the bottom piece as there is no right piece
                if index + 1 <= self.cols * (self.rows - 1):
                    proximity_value = self.compare_overall_values(solution[index].overall_value,
                                                                  solution[index + self.cols].overall_value)
                    color_compared = color_compared + proximity_value

        return color_compared

    def compare_edge_values(self, edge_one: list, edge_two: list) -> int:
        return abs(edge_one[0] - edge_two[0] + edge_one[1] - edge_two[1] + edge_one[2] - edge_two[2])

    def compare_overall_values(self, value_one, value_two):
        return abs(value_one[0] - value_two[0] + value_one[1] - value_two[1] + value_one[2] - value_two[2])


class Evolution:
    """
    class Evolution
    Mutation function (pieces)
    - exchanges x pieces randomly

    Starting shuffler (pieces)
    - randomly shuffles the pieces, creating new solutions
    """


    @staticmethod
    def mutate(solution: list, mutation_tolerance: int):  # mutates the element by switching two of the pieces
        mutated = solution[:]  # TODO look into improving the mutate method by crossing over or make a new method for crossing over
        for _ in range(random.randint(1, mutation_tolerance)):
            a_index = random.randint(0, len(mutated) - 1)
            b_index = random.randint(0, len(mutated) - 1)
            temp = mutated[b_index]
            mutated[b_index] = mutated[a_index]
            mutated[a_index] = temp
        return mutated

    @staticmethod
    def create_new_solutions(pieces: list, num_of_solutions: int):
        new_solutions = []
        for _ in range(num_of_solutions):
            new_solutions.append(sorted(pieces, key=lambda x: random.random()))
        return new_solutions


class GeneticAlgorithm:
    """
    class GeneticAlgorithm

    start everything and manage everything

    init
    params:
    - how many things per generation
    - rows, cols
    - image to work on
    """

    image = []
    rows = -1
    cols = -1
    generation_size = 200
    generation_num = 500
    pieces = []
    img_height = -1
    img_width = -1
    row_size = -1
    col_size = -1
    edge_threshold = 20

    def __init__(self, image, rows, cols, sol_num_per_generation, generation_num, edge_threshold):
        self.image = image
        self.rows = rows
        self.cols = cols
        self.generation_size = sol_num_per_generation
        self.img_width = image.shape[1]
        self.img_height = image.shape[0]
        self.row_size = self.img_height // self.rows
        self.col_size = self.img_width // self.cols
        self.generation_num = generation_num
        self.edge_threshold = edge_threshold

        self.pieces = self.make_pieces()

    def make_image(self, solution):  # returns an image made from the pieces given
        all_rows = []
        for i in range(self.rows):
            row_pieces = solution[i * self.cols:(i + 1) * self.cols]
            row_images = []
            for piece in row_pieces:
                row_images.append(piece.whole_piece)
            row = np.concatenate(row_images, axis=1)
            all_rows.append(row)

        grid = np.concatenate(all_rows)
        return grid

    def make_pieces(self):  # gets the pieces from the image
        pieces = []
        for i in range(self.rows):
            for j in range(self.cols):
                pieces.append(Piece(i * self.rows + j, self.image[i * self.row_size:(i + 1) * self.row_size,
                                                       j * self.col_size:(j + 1) * self.col_size]))
        return pieces

    def start_generation(self):  # main functionality
        start_solutions = Evolution.create_new_solutions(self.pieces, 1000)
        last_solution = 999999999
        same_num = 0
        solutions = start_solutions
        top_solution = None
        for gen in range(
                self.generation_num):  # genetic algorithm. Gets the x best solutions, mutates them and checks which ones are the best. Repeats
            curr_generation = Generation(solutions, 100, self.rows, self.cols, self.edge_threshold)
            ranked = curr_generation.get_ranked_solutions()
            top_solution = ranked[0]
            print(f"Generation {gen} best solution: \n {top_solution[0]}")
            if top_solution[0] == last_solution:
                same_num = same_num + 1
            else:
                same_num = 0
                last_solution = top_solution[0]
            # if same_num > 500:
            #     print("No changes for over 500 generations! \nEnding the program")
            #     break

            new_solutions = []
            mutated_weight = 0.8
            mutated_num = math.floor(self.generation_size * mutated_weight)
            mutation_tolerance = self.generation_num // 5
            mutation_tolerance = mutation_tolerance + same_num
            for j in range(mutated_num):
                new_solutions.append(
                    Evolution.mutate(ranked[random.randint(0, 99)][1], max(1, gen // mutation_tolerance)))

            new_solutions.extend(Evolution.create_new_solutions(self.pieces, self.generation_size - mutated_num))
            for ranked_sol in ranked:
                new_solutions.append(ranked_sol[1])
            solutions = new_solutions

        final_image = self.make_image(top_solution[1])
        return final_image


IMAGE = 'black_white_test.png'
img = cv.imread("scrambledImages/" + IMAGE, cv.IMREAD_COLOR)

start_timestamp = datetime.datetime.now()
# 10, 10, 200, 500, 20
algorithm = GeneticAlgorithm(img, rows=9, cols=8, sol_num_per_generation=1000, generation_num=2000, edge_threshold=20)
final_img = algorithm.start_generation()
curr_timestamp = datetime.datetime.now()

print(f"Start: {start_timestamp}\nEnd: {curr_timestamp}")

scale_down_factor = 1
final_img = cv.resize(final_img, (math.floor(final_img.shape[1] * scale_down_factor), math.floor(final_img.shape[0] * scale_down_factor)))
cv.imshow('test', final_img)
cv.imwrite("unscrambledImages/" + IMAGE, final_img)
cv.waitKey(0)
cv.destroyAllWindows()
