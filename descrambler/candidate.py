from descrambler.piece import Piece


class Candidate:
    piece = Piece
    fitness_value = float

    def __init__(self, piece: Piece, fitness_value: float):
        self.piece = piece
        self.fitness_value = fitness_value
