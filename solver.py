import cv2 as cv
import numpy as np

class Piece:
    top_edge = []
    bottom_edge = []
    right_edge = []
    left_edge = []
    whole_piece =[]
    index = -1

    def __init__(self, index, piece):
        self.whole_piece = piece
        self.index = index
        piece = cv.cvtColor(piece, cv.COLOR_BGR2GRAY)
        self.top_edge = piece[0]
        self.bottom_edge = piece[len(piece)-1]
        self.right_edge = piece[:, len(piece[0])-1]
        self.left_edge = piece[:, 0]

class Generation:

    """
    init
    parameters: top solutions from previous gen, how many solutions to return in fitness


    fitness function
        compare edges
            returns likelihood of the squares being close by checking how far away the colors are from each other
            always divide the result by the number of corners since somewhere we only look at one corner
        sort
        returns top x solutions in format (quality_value, solution)
    """

"""
class Evolution? //maybe change name later
Mutation function (pieces)
- exchanges x pieces randomly

Starting shuffler (pieces)
- randomly shuffles the pieces, creating new solutions
"""


IMAGE = 'test.jpg'

img = cv.imread("scrambledImages/" + IMAGE, cv.IMREAD_COLOR)
width = img.shape[1]
height = img.shape[0]

rows = 10
cols = 10

row_size = height // rows
col_size = width // cols

pieces = []

for i in range(rows):
    for j in range(cols):
        pieces.append(Piece(i*rows + j, img[i * row_size:(i + 1) * row_size, j * col_size:(j + 1) * col_size]))




