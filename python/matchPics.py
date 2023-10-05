import cv2
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
# additional imports

# Q2.1.4

def matchPics(I1, I2, opts):
        """
        Match features across images

        Input
        -----
        I1, I2: Source images
        opts: Command line args

        Returns
        -------
        matches: List of indices of matched features across I1, I2 [p x 2]
        locs1, locs2: Pixel coordinates of matches [N x 2]
        """
        
        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
        

        # TODO: Convert Images to GrayScale
        I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
        
        # TODO: Detect Features in Both Images
        I1_loc = corner_detection(I1_gray, sigma)
        I2_loc = corner_detection(I2_gray, sigma)
        
        print('found corners')
        # TODO: Obtain descriptors for the computed feature locations
        I1_desc, I1_loc = computeBrief(I1_gray, I1_loc)
        I2_desc, I2_loc = computeBrief(I2_gray, I2_loc)
        print('found descriptors')

        # TODO: Match features using the descriptors
        matches = briefMatch(I1_desc, I2_desc, ratio)
        
        return matches, I1_loc, I2_loc


def matchPicsV2(I1, I2_desc, I2_loc, opts):
        # TODO: Convert Images to GrayScale
        I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        
        # TODO: Detect Features in Both Images
        I1_loc = corner_detection(I1_gray, opts.sigma)
        
        # TODO: Obtain descriptors for the computed feature locations
        I1_desc, I1_loc = computeBrief(I1_gray, I1_loc)

        # TODO: Match features using the descriptors
        matches = briefMatch(I1_desc, I2_desc, opts.ratio)
        
        return matches, I1_loc, I2_loc