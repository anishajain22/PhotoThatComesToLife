import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

# Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH


# Q2.2.4

def warpImage(opts):
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')
    image3 = cv2.imread('../data/hp_cover.jpg')

    matches, locs1, locs2 = matchPics(image2, image1, opts)

    locs1_matches = locs1[matches[:,0]]
    locs2_matches = locs2[matches[:,1]]
    locs1_matches[:,[0, 1]] = locs1_matches[:,[1, 0]]
    locs2_matches[:,[0, 1]] = locs2_matches[:,[1, 0]]

    H2to1, _ = computeH_ransac(locs1_matches, locs2_matches, opts)

    # output = cv2.warpPerspective(image3, H2to1, (image2.shape[1], image2.shape[0]), flags=cv2.INTER_LINEAR)
    
    target_height, target_width = image1.shape[:2]
    resized_image3 = cv2.resize(image3, (target_width, target_height))
    output = compositeH(H2to1, resized_image3, image2)
    cv2.imwrite("output.jpg", output)

    pass

if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


