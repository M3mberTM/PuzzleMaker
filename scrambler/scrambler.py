import cv2 as cv
import random

import numpy as np


def shuffle_image(img, rows, cols):
    pieces = []
    width = img.shape[1]
    height = img.shape[0]
    row_size = height // rows
    col_size = width // cols

    if height % rows != 0:
        print('Rows don\'t fully match')
        img = img[0:len(img)-(height%rows)]
    if width % cols != 0:
        print('Cols don\'t fully match')
        img = img[:, 0: len(img[0])-(width%cols)]

    for i in range(rows):
        for j in range(cols):
            pieces.append(img[i * row_size:(i + 1) * row_size, j * col_size:(j + 1) * col_size])

    random.shuffle(pieces)
    all_rows = []
    for i in range(rows):
        row_images = pieces[i * cols:(i + 1) * cols]
        row = np.concatenate(row_images, axis=1)
        all_rows.append(row)

    grid = np.concatenate(all_rows)
    return grid



IMAGE = "test_colorEdges.jpg"
rows = 10
cols = 10
scale_down_factor = 2

img = cv.imread("testImages/" + IMAGE, cv.IMREAD_COLOR)
img = cv.resize(img, (img.shape[1] // scale_down_factor, img.shape[0] // scale_down_factor))
grid = shuffle_image(img, rows, cols)

cv.imshow('test', img)
cv.imshow('test_scrambled', grid)
cv.imwrite("scrambledImages/" + IMAGE, grid)
cv.waitKey(0)
cv.destroyAllWindows()
