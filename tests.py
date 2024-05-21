import cv2 as cv
import numpy as np

def show_image_edges_vertical(image_num, image_two_num):
    image_two = cv.imread("pieces/" + str(image_two_num) + ".jpg", cv.IMREAD_COLOR)
    image_two = cv.cvtColor(image_two, cv.COLOR_BGR2GRAY)
    image_one = cv.imread("pieces/" + str(image_num) + ".jpg", cv.IMREAD_COLOR)

    image_one = cv.cvtColor(image_one, cv.COLOR_BGR2GRAY)


    final_img = np.concatenate([image_one, image_two])
    edge_fit_blur = cv.GaussianBlur(final_img, (3, 3), 0)  # image blur
    edges = cv.Canny(image=edge_fit_blur, threshold1=30, threshold2=75)  # Canny Edge Detection

    edges_height = edges.shape[0]
    final_edge = np.array(edges[(edges_height // 2) - 1:(edges_height // 2) + 1])
    edge_length = final_edge.size
    edge_sum = np.sum(final_edge)
    print(edge_sum / edge_length)
    print("--------------------")
    cv.imshow("final_edge", final_edge)
    cv.imshow("image", final_img)
    cv.imshow("edges",edges)
    cv.waitKey(0)


def show_image_edges_horizontal(image_num, image_two_num):
    image_two = cv.imread("pieces/" + str(image_two_num) + ".jpg", cv.IMREAD_COLOR)
    image_two = cv.cvtColor(image_two, cv.COLOR_BGR2GRAY)
    image_one = cv.imread("pieces/" + str(image_num) + ".jpg", cv.IMREAD_COLOR)

    image_one = cv.cvtColor(image_one, cv.COLOR_BGR2GRAY)


    final_img = np.concatenate([image_one, image_two], axis=1)
    edge_fit_blur = cv.GaussianBlur(final_img, (3, 3), 0)  # image blur
    edges = cv.Canny(image=edge_fit_blur, threshold1=30, threshold2=75)  # Canny Edge Detection

    edges_width = edges.shape[1]
    final_edge = np.array(edges[:, (edges_width // 2) - 1:(edges_width // 2) + 1])
    edge_length = final_edge.size
    edge_sum = np.sum(final_edge)
    print(edge_sum / edge_length)
    print("--------------------")
    cv.imshow("final_edge", final_edge)
    cv.imshow("image", final_img)
    cv.imshow("edges",edges)
    cv.waitKey(0)

print("PROGRAM STARTED")

show_image_edges_vertical(44, 25)
show_image_edges_horizontal(60, 1)
