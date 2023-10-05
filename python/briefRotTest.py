import numpy as np
import cv2
from matchPics import matchPics
from scipy import ndimage, datasets
from opts import get_opts
import matplotlib.pyplot as plt
from displayMatch import displayMatched
from scipy.ndimage import sobel
from skimage.transform import rotate
from helper import computeBrief, corner_detection, briefMatch

#Q2.1.6

def rotTest(opts):

    # TODO: Read the image and convert to grayscale, if necessary
    img = cv2.imread('../data/cv_cover.jpg')
    bins = []
    matches_count = []
    for i in range(36):
        # TODO: Rotate Image
        img_rot = ndimage.rotate(img, i*10, reshape=False)

        # TODO: Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(img, img_rot, opts)

        # TODO: Update histogram
        bins.append(i * 10)  # Store the rotation angle
        matches_count.append(len(matches))
        
        if i == 1 or i == 10 or i == 27:
            displayMatched(opts, img, img_rot)
    # TODO: Display histogram
    plt.bar(bins, matches_count, width=10)  # Adjust width as needed
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Number of Matches')
    plt.title('Histogram of Matches vs. Rotation Angle')
    plt.show()

    # Perform feature matching based on descriptor and orientation similarity
    matches = []

    for i in range(len(desc1)):
        min_dist1 = float('inf')
        min_dist2 = float('inf')
        match_idx = -1

        for j in range(len(desc2)):
            # Check orientation similarity
            orientation_diff = np.abs(orientations1[i] - orientations2[j])
            if orientation_diff > np.pi:
                orientation_diff = 2 * np.pi - orientation_diff
            
            # Compute Hamming distance
            dist = hammingDistance(desc1[i], desc2[j])

            # Check if it's a good match
            if dist < min_dist1:
                min_dist2 = min_dist1
                min_dist1 = dist
                match_idx = j
            elif dist < min_dist2:
                min_dist2 = dist

        # Ratio test for feature matching
        if min_dist1 < ratio * min_dist2:
            matches.append((i, match_idx))

    return matches


if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
