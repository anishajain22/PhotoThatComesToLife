import numpy as np
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt
import skimage.feature
from scipy import ndimage

PATCHWIDTH = 9
NBITS = 256

def briefMatch(desc1,desc2,ratio):
    matches = skimage.feature.match_descriptors(desc1,desc2,'hamming',cross_check=True,max_ratio=ratio)
    return matches

def plotMatches(im1,im2,matches,locs1,locs2):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.axis('off')
    skimage.feature.plot_matches(ax,im1,im2,locs1,locs2,matches,matches_color='r',only_matches=True)
    plt.show()
    return

def makeTestPattern(seed=0):
    np.random.seed(seed)
    compareX = np.floor(PATCHWIDTH * PATCHWIDTH * np.random.random((NBITS, 1))).astype(int)
    compareY = np.floor(PATCHWIDTH * PATCHWIDTH * np.random.random((NBITS, 1))).astype(int)
    return compareX, compareY

def computePixel(img, idx1, idx2, width, center):
    halfWidth = width // 2
    col1 = idx1 % width - halfWidth
    row1 = idx1 // width - halfWidth
    col2 = idx2 % width - halfWidth
    row2 = idx2 // width - halfWidth
    return 1 if img[int(center[0]+row1)][int(center[1]+col1)] < img[int(center[0]+row2)][int(center[1]+col2)] else 0

def computeBrief(img, locs):
    patchWidth = 9
    nbits = 256
    compareX, compareY = makeTestPattern()
    m, n = img.shape

    halfWidth = patchWidth//2

    locs = np.array(list(filter(lambda x: halfWidth <= x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))
    desc = np.array([list(map(lambda x: computePixel(img, x[0], x[1], patchWidth, c), zip(compareX, compareY))) for c in locs])
    return desc, locs

def computePixelVectorized(img, compareX, compareY, patchWidth, center):
    halfWidth = patchWidth // 2
    col1 = compareX % patchWidth - halfWidth
    row1 = compareX // patchWidth - halfWidth
    col2 = compareY % patchWidth - halfWidth
    row2 = compareY // patchWidth - halfWidth
    
    pixel_values1 = img[np.round(center[0] + row1).astype(int), np.round(center[1] + col1).astype(int)]
    pixel_values2 = img[np.round(center[0] + row2).astype(int), np.round(center[1] + col2).astype(int)]
    
    return (pixel_values1 < pixel_values2).astype(int).flatten()

# def computeBrief(img, locs):
#     m, n = img.shape
#     halfWidth = PATCHWIDTH // 2
#     compareX, compareY = makeTestPattern()
#     locs = np.array(list(filter(lambda x: halfWidth <= x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))
#     desc = np.array([computePixelVectorized(img, compareX, compareY, PATCHWIDTH, center) for center in locs])
#     return desc, locs

def corner_detection(img, sigma):

    # fast method
    result_img = skimage.feature.corner_fast(img, n=PATCHWIDTH, threshold=sigma)
    locs = skimage.feature.corner_peaks(result_img, min_distance=1)
    return locs


def loadVid(path):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    cap = cv2.VideoCapture(path)

    # Append frames to list
    frames = []

    # Check if camera opened successfully
    if cap.isOpened()== False:
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            #Store the resulting frame
            frames.append(frame)
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    frames = np.stack(frames)

    return frames
