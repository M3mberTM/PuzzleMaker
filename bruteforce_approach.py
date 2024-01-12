import cv2 as cv
import random
import numpy as np
import datetime


def make_image(pieces, rows, cols):
    all_rows = []
    for i in range(rows):
        row_images = pieces[i * cols:(i + 1) * cols]
        row = np.concatenate(row_images, axis=1)
        all_rows.append(row)

    grid = np.concatenate(all_rows)
    return grid


def fitness(pieces, rows, cols):
    image = make_image(pieces, rows, cols)

    return get_edges(cv.cvtColor(image, cv.COLOR_BGR2GRAY), rows, cols)
def get_edges(image, rows, cols):
    row_edges = []
    col_edges = []
    for i in range(1, rows):
        row_edges.append(image[i * row_size:i * row_size + 2, 0:width])

    for i in range(1, cols):
        col_edges.append(image[0:height, i * col_size:i * col_size + 2])

    total_weight = 0
    for i in range(len(row_edges)):
        edges = compare_edges(row_edges[i], 20)
        total_weight = total_weight + edges

    for i in range(len(col_edges)):
        edges = compare_edges(cv.rotate(col_edges[i], cv.ROTATE_90_CLOCKWISE), 20)
        total_weight = total_weight + edges

    return total_weight


def compare_edges(image, treshold: int) -> int:
    top_edge = image[0]
    bottom_edge = image[1]
    edge_likelihood = 0
    for i in range(len(top_edge)):
        top_pixel = top_edge[i]
        bottom_pixel = bottom_edge[i]
        difference = top_pixel-bottom_pixel
        if abs(difference) > treshold:
            edge_likelihood = edge_likelihood + 1

    return edge_likelihood

def compare_colors(pieces, rows, cols):
    all_rows = []
    for i in range(rows):
        row_images = pieces[i * cols:(i + 1) * cols]
        all_rows.append(row_images)
    overall_deviation = 0
    for row in range(len(all_rows)):
        for col in range(len(all_rows[row])):
            piece = all_rows[row][col]
            neighbors = get_neighbors(row, col, rows, cols)
            piece_color_val = np.sum(piece)
            expected_color_val = piece_color_val * len(neighbors)
            real_color_val = 0
            for neighbor in neighbors:
                real_color_val = real_color_val + np.sum(all_rows[neighbor[0]][neighbor[1]])
            overall_deviation = overall_deviation + abs(expected_color_val - real_color_val)

    return overall_deviation




def get_neighbors(row, col, max_rows, max_cols):
    neighbors_row = [-1, 0, 1]
    neighbors_col = [-1, 0, 1]
    all_neighbors = []
    for r_neighbor in neighbors_row:
        for c_neighbor in neighbors_col:
            neighbor = [r_neighbor + row, c_neighbor + col]
            if is_valid_neighbor(r_neighbor + row, c_neighbor + col, max_rows, max_cols) and not (
                    r_neighbor == 0 and c_neighbor == 0):
                all_neighbors.append(neighbor)
    return all_neighbors


def is_valid_neighbor(row, col, max_rows, max_cols) -> bool:
    if row > max_rows - 1:
        return False
    if row < 0:
        return False
    if col > max_cols - 1:
        return False
    if col < 0:
        return False
    return True


def get_grid_likelihood(image, rows, cols):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    width = image.shape[1]
    height = image.shape[0]

    corners = cv.goodFeaturesToTrack(gray, rows * cols, 0.0001, min(width//cols, height//rows))
    corners = np.intp(corners)
    x_values = []
    y_values = []
    for corner in corners:
        x, y = corner.ravel()
        x_values.append(x)
        y_values.append(y)

    overall_value = 0
    for value in x_values:
        overall_value = overall_value + x_values.count(value)

    for value in y_values:
        overall_value = overall_value + y_values.count(value)
    return overall_value


def get_contour_num(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_num = len(contours)

    return contour_num


def mutate(og):
    pieces = og[:]
    a_index = random.randint(0, len(pieces) - 1)
    b_index = random.randint(0, len(pieces) - 1)
    # switch those two elements in place
    temp = pieces[b_index]
    pieces[b_index] = pieces[a_index]
    pieces[a_index] = temp
    return pieces


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
        pieces.append(img[i * row_size:(i + 1) * row_size, j * col_size:(j + 1) * col_size])

# Generating solutions
solutions = []
for i in range(1000):
    solutions.append(sorted(pieces, key=lambda x: random.random()))

last_solution = 999999999
same_num = 0
start_timestamp = datetime.datetime.now()
for i in range(500):
    ranked = []
    for solution in solutions:
        ranked.append((fitness(solution, rows, cols), solution))
    ranked.sort(key=lambda x: x[0])
    print(f"Generation {i} best solution")
    top_solution = ranked[0][0]
    print(top_solution)
    if top_solution == last_solution:
        same_num = same_num + 1
    else:
        same_num = 0
        last_solution = top_solution
    if same_num > 100:
        break

    best_solutions = ranked[:100]

    new_generation = []
    for j in range(100):
        new_generation.append(mutate(best_solutions[random.randint(0, 99)][1]))
    for sol in best_solutions:
        new_generation.append(sol[1])

    solutions = new_generation

img = make_image(best_solutions[0][1], rows, cols)
curr_timestamp = datetime.datetime.now()
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# gray_corners = cv.goodFeaturesToTrack(gray, rows * cols, 0.0001, max(width//cols, height//rows))
# gray_corners = np.intp(gray_corners)
#
# for corner in gray_corners:
#     x, y = corner.ravel()
#     cv.circle(img, (x, y), 5, (0, 255, 0), -1)

print(f"Start: {start_timestamp}\nEnd: {curr_timestamp}")
cv.imshow('test', img)
cv.imwrite("unscrambledImages/" + IMAGE, img)
cv.waitKey(0)
cv.destroyAllWindows()
