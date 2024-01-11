import cv2 as cv
import numpy as np
from functools import cmp_to_key

"""
IDEA

find contours and use them to find all the pieces for the puzzle
puzzle solving will use probably neural networks, but very simple
thing to take into account
- overall piece color
- compare the edge colors
(put them slowly into the algorithm and watch how much of a difference it makes!!!)

"""

def get_similarity(piece: list, piece_two: list) -> int:
    return np.sum(np.absolute((piece - piece_two)))

def get_edges(pieces):
    edges = {}
    for key in pieces.keys():
        piece = pieces[key]
        top_edge = piece[0]
        bottom_edge = piece[len(piece) - 1]
        left_edge = piece[:, 0, :]
        right_edge = piece[:, len(piece[0]) - 1, :]
        edges[key] = {'top': top_edge, 'bottom': bottom_edge, 'right': right_edge, 'left': left_edge}
    return edges

def sort_by_edges(similarity_obj):
    for main_piece in similarity_obj.keys():
        for side in similarity_obj[main_piece].keys():
            similarity_obj[main_piece][side].sort(key=cmp_to_key(object_value_sort))

def object_value_sort(a, b):
    a_value = list(a.values())[0]
    b_value = list(b.values())[0]
    if a_value > b_value:
        return 1
    elif a_value == b_value:
        return 0
    else:
        return -1


def get_best_next_piece(similarity_obj, last_piece, upper_piece, row, col, ignore_pieces):

    if row == 0:
        left_piece_sim = similarity_obj[last_piece]['right']
        next_piece = -1
        for piece in left_piece_sim:
            if list(piece.keys())[0] not in ignore_pieces:
                return list(piece.keys())[0]
        return next_piece
    else:
        upper_piece_sim = similarity_obj[upper_piece]['bottom']
        if col == 0:
            next_piece = -1
            for piece in upper_piece_sim:
                if list(piece.keys())[0] not in ignore_pieces:
                    return list(piece.keys())[0]
            return next_piece
        else:
            left_piece_sim = similarity_obj[last_piece]['right']
            curr_sim = 9999999999999999
            next_piece = -1
            for piece in upper_piece_sim:
                key = list(piece.keys())[0]
                if key in ignore_pieces:
                    continue
                for l_piece in left_piece_sim:
                    if list(l_piece.keys())[0] == key:
                        if piece[key] + l_piece[key] < curr_sim:
                            curr_sim = piece[key] + l_piece[key]
                            next_piece = key
            return next_piece


# TODO solver function
IMAGE = 'test3.jpg'

img = cv.imread("scrambledImages/" + IMAGE, cv.IMREAD_COLOR)
width = img.shape[1]
height = img.shape[0]

rows = 10
cols = 10

row_size = height // rows
col_size = width // cols

pieces = {}

for i in range(rows):
    for j in range(cols):
        pieces[str(i*rows + j)] = img[i * row_size:(i + 1) * row_size, j * col_size:(j + 1) * col_size]

edges = get_edges(pieces)

similar = {}
for key in pieces.keys():
    similar[key] = {'top': [], 'bottom': [], 'right': [], 'left': []}
    for key_two in pieces.keys():
        if key != key_two:
            top_similarity = get_similarity(edges[key]['top'], edges[key_two]['top'])
            bottom_similarity = get_similarity(edges[key]['bottom'], edges[key_two]['bottom'])
            right_similarity = get_similarity(edges[key]['right'], edges[key_two]['right'])
            left_similarity = get_similarity(edges[key]['left'], edges[key_two]['left'])
            similar[key]['top'].append({key_two: top_similarity})
            similar[key]['right'].append({key_two: right_similarity})
            similar[key]['left'].append({key_two: left_similarity})
            similar[key]['bottom'].append({key_two: bottom_similarity})


sort_by_edges(similar)


start_piece = '0'
ignore_pieces = [start_piece]
# TODO putting everything together
final_img = []
for curr_row in range(rows):
    row = []
    for curr_col in range(cols):
        if curr_col + curr_row == 0:
            row.append(start_piece)
        else:
            if curr_row == 0:
                next = get_best_next_piece(similar, row[len(row)-1], None, curr_row, curr_col, ignore_pieces)
                row.append(next)
                ignore_pieces.append(next)
            else:
                if curr_col == 0:
                    next = get_best_next_piece(similar, None, final_img[curr_row-1][curr_col], curr_row, curr_col, ignore_pieces)
                else:
                    next = get_best_next_piece(similar, row[len(row)-1], final_img[curr_row-1][curr_col], curr_row, curr_col, ignore_pieces)
                row.append(next)
                ignore_pieces.append(next)
    final_img.append(row)


all_rows = []
for row in final_img:
    row_images = []
    for col in row:
        row_images.append(pieces[col])
    all_rows.append(np.concatenate(row_images, axis=1))

grid = np.concatenate(all_rows)
print(grid)



cv.imshow('test', img)
cv.imshow('finished', grid)
cv.waitKey(0)
cv.destroyAllWindows()
