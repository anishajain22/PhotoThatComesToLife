from multiprocessing import Pool
import multiprocessing
import numpy as np
import cv2
import time
import os
from opts import get_opts
from matchPics import matchPics, matchPicsV2
from planarH import computeH_ransac, compositeH
from helper import loadVid
import concurrent.futures
import matplotlib.pyplot as plt

from helper import computeBrief, corner_detection

def remove_black_border(frames):
    # Convert the image to grayscale
    gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    # Find the rows with non-zero values (non-black rows)
    non_black_rows = np.where(np.any(gray != 0, axis=1))[0]

    # Determine the top and bottom rows of the non-black region
    top_row = non_black_rows[0]
    bottom_row = non_black_rows[-1]

    cropped_frames = []
    for frame in frames:
        # Crop the frame to match the maximum ROI dimensions
        cropped_frame = frame[top_row:bottom_row + 1, :]
        cropped_frames.append(cropped_frame)

    return cropped_frames

# crop image
def crop_image(source_image, target_image):
    # Get the dimensions (width and height) of the target image
    target_height, target_width = target_image.shape[:2]

    # Calculate the cropping region for the source image
    source_height, source_width = source_image.shape[:2]
    crop_x = max(0, (source_width - target_width) // 2)
    crop_y = max(0, (source_height - target_height) // 2)

    # Crop the source image to match the target image dimensions
    cropped_image = source_image[crop_y:crop_y + target_height, crop_x:crop_x + target_width]
    if cropped_image.shape[:2] != (target_height, target_width):
        cropped_image = cv2.resize(cropped_image, (target_width, target_height))

    return cropped_image

# warp_frames
def warp_frames(base_image, x1_frames, x2_frames):
    output_frames = []
    for i in range(min(len(x1_frames), len(x2_frames))):
        matches, locs1, locs2 = matchPics(x1_frames[i], base_image, opts)
        if len(matches) < 4:
            continue
        locs1_matches = locs1[matches[:,0]]
        locs2_matches = locs2[matches[:,1]]
        locs1_matches[:,[0, 1]] = locs1_matches[:,[1, 0]]
        locs2_matches[:,[0, 1]] = locs2_matches[:,[1, 0]]
        H2to1, _ = computeH_ransac(locs1_matches, locs2_matches, opts)
        cropped_image = crop_image(x2_frames[i], base_image)
        output_frames.append(compositeH(H2to1, cropped_image, x1_frames[i]))
    return output_frames

def create_video(frames, video_path, frame_rate = 10):
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec for MP4 format
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
    for frame in frames:
        out.write(frame)
        out.release()
    print(f'Video saved as {video_path}')

if __name__ == "__main__":

    opts = get_opts()
    x1_frames = loadVid('../data/book.mov')
    x2_frames = loadVid('../data/ar_source.mov')
    x2_frames =  remove_black_border(x2_frames)
    
    base_image = cv2.imread('../data/cv_cover.jpg')
    
    output_frames = warp_frames(base_image, x1_frames, x2_frames)
    create_video(output_frames, 'output.avi', opts.fps)

    cv2.destroyAllWindows()