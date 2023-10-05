import numpy as np
import cv2
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from planarH import applyH
# Q4

def panaroma_image(opts):
    image1 = cv2.imread('../data/cathedral_left.png')
    image2 = cv2.imread('../data/cathedral_right.png')

    matches, locs1, locs2 = matchPics(image2, image1, opts)

    locs1_matches = locs1[matches[:,0]]
    locs2_matches = locs2[matches[:,1]]
    locs1_matches[:,[0, 1]] = locs1_matches[:,[1, 0]]
    locs2_matches[:,[0, 1]] = locs2_matches[:,[1, 0]]

    H1to2, _ = computeH_ransac(locs2_matches, locs1_matches, opts)
    height, width = image1.shape[:2]
    
    width_of_warped_image = get_width_of_warped_points(height, width, H1to2)
    
    warped_image = cv2.warpPerspective(image2, H1to2, (width + width_of_warped_image, height), flags=cv2.INTER_LINEAR)
    warped_image[:, :width] = image1
    warped_image = remove_black_columns(warped_image)
    cv2.imwrite("output.png", warped_image)

def remove_black_columns(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the column sums (sum of pixel values) for each column
    column_sums = np.sum(gray, axis=0)

    # Find the first and last non-zero column indices
    left_index = 0
    right_index = len(column_sums) - 1

    for i, sum_value in enumerate(column_sums):
        if sum_value > 0:
            left_index = i
            break

    for i in range(len(column_sums) - 1, -1, -1):
        if column_sums[i] > 0:
            right_index = i
            break

    # Crop the image to remove the black border
    img_cropped = image[:, left_index:right_index + 1]
    return img_cropped

def get_width_of_warped_points(height, width, H1to2):
    bottom_left = (0, 0)
    bottom_right = (width, 0)
    top_left = (0, height)
    top_right = (width, height)
    x = []
    x.append(bottom_left)
    x.append(bottom_right)
    x.append(top_left)
    x.append(top_right)
    x = np.array(x)

    transformed_points = applyH(H1to2, x)

    return int(transformed_points[1][0] - transformed_points[0][0])
if __name__ == "__main__":

    opts = get_opts()
    panaroma_image(opts)
