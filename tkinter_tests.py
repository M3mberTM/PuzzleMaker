import cv2 as cv
import numpy as np


def get_right_edge(image_two_num):
    image_two = cv.imread("pieces/" + str(image_two_num) + ".jpg", cv.IMREAD_COLOR)
    image_two = cv.cvtColor(image_two, cv.COLOR_BGR2GRAY)
    best_match = (-1, 3000)
    for i in range(0, 72):
        print("-----IMAGE NUMERO:" + str(i) + "-----")
        image_one = cv.imread("pieces/" + str(i) + ".jpg", cv.IMREAD_COLOR)

        image_one = cv.cvtColor(image_one, cv.COLOR_BGR2GRAY)


        final_img = np.concatenate([image_one, image_two], axis=1)
        edge_fit_blur = cv.GaussianBlur(final_img, (3, 3), 0)  # image blur
        edges = cv.Canny(image=edge_fit_blur, threshold1=80, threshold2=140)  # Canny Edge Detection

        edges_width = edges.shape[1]
        final_edge = np.array(edges[:, (edges_width // 2) - 1:(edges_width // 2) + 1])
        edge_length = final_edge.size
        edge_sum = np.sum(final_edge)
        print(edge_sum / edge_length)
        print("--------------------")
        if best_match[1] > edge_sum / edge_length:
            best_match = (i, edge_sum / edge_length)
    return best_match



print("PROGRAM START...")

print("LOADING PIECES")


print("TESTS")
edge = get_right_edge(1)[0]
assert edge == 60, f"Num: 1. Expected 60, got {edge}"
edge = get_right_edge(60)[0]
assert edge == 25, f"Num: 60. Expected 25, got {edge}"
edge = get_right_edge(49)[0]
assert edge == 15, f"Num: 49.Expected 15, got {edge}"
edge = get_right_edge(50)[0]
assert edge == 63, f"Num: 50. Expected 63, got {edge}"
edge = get_right_edge(0)[0]
assert edge == 50, f"Num: 50. Expected 50, got {edge}"

print("TESTS PASSED")

# cv.imshow("final_edge", final_edge)
# cv.imshow("image", final_img)
# cv.imshow("edges",edges)
# cv.waitKey(0)
