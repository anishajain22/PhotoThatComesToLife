# import numpy as np
# import cv2


# def computeH(x1, x2):
#     #Q2.2.1
#     # TODO: Compute the homography between two sets of points
#     A = []
#     for i in range(len(x1)):
#         x, y = x1[i][0], x1[i][1]
#         u, v = x2[i][0], x2[i][1]
#         A.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
#         A.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])

#     A = np.array(A)
#     U, S, Vt = np.linalg.svd(A)
#     H2to1 = Vt[-1, :].reshape(3, 3)
#     return H2to1


# def computeH_norm(x1, x2):
#     #Q2.2.2
#     # TODO: Compute the centroid of the points
#     x1_centroid = np.mean(x1, axis = 0)
#     x2_centroid = np.mean(x2, axis=0)

#     # TODO: Shift the origin of the points to the centroid
#     x1_shifted = x1 - x1_centroid
#     x2_shifted = x2 - x2_centroid

#     # TODO: Normalize the points so that the largest distance from the origin is equal to sqrt(2)
#     x1_max_dist = np.max(np.linalg.norm(x1_shifted, axis=1))
#     x2_max_dist = np.max(np.linalg.norm(x2_shifted, axis=1))
    
#     x1_scale = np.sqrt(2) / x1_max_dist
#     x2_scale = np.sqrt(2) / x2_max_dist
    
#     x1_normalized = x1_shifted * x1_scale
#     x2_normalized = x2_shifted * x2_scale

#     # TODO: Similarity transform 1 (scaling and translation)
#     T1 = np.array([[x1_scale, 0, -x1_scale * x1_centroid[0]],
#                    [0, x1_scale, -x1_scale * x1_centroid[1]],
#                    [0, 0, 1]])

#     # TODO: Similarity transform 2 (scaling and translation)
#     T2 = np.array([[x2_scale, 0, -x2_scale * x2_centroid[0]],
#                    [0, x2_scale, -x2_scale * x2_centroid[1]],
#                    [0, 0, 1]])

#     # TODO: Compute homography
#     H2to1 = computeH(x1_normalized, x2_normalized)

#     # TODO: Denormalization
#     H2to1 = np.linalg.inv(T1).dot(H2to1).dot(T2)

#     return H2to1

# def computeH_ransac(locs1, locs2, opts):
#     #Q2.2.3
#     #Compute the best fitting homography given a list of matching points
#     max_iters = opts.max_iters  # the number of iterations to run RANSAC for
#     inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

#     bestH2to1 = None
#     best_inliers = []

#     for _ in range(max_iters):
#         # Randomly select 4 point correspondences
#         random_indices = np.random.choice(locs1.shape[0], 4, replace=False)
#         random_locs1 = locs1[random_indices]
#         random_locs2 = locs2[random_indices]

#         # Compute the homography based on the random points
#         H2to1 = computeH_norm(random_locs1, random_locs2)
#         transform_locs2 = applyH(H2to1, locs2)

#         # Caluclate euclidean distances
#         distances = np.linalg.norm(locs1 - transform_locs2, axis=1)

#         # inliers
#         inliers = distances < inlier_tol
#         if np.sum(inliers) > np.sum(best_inliers):
#             best_inliers = inliers
#             bestH2to1 = H2to1

#     return bestH2to1, best_inliers

# def compositeH(H2to1, template, img):
    
#     #Create a composite image after warping the template image on top
#     #of the image using the homography

#     #Note that the homography we compute is from the image to the template;
#     #x_template = H2to1*x_photo
#     #For warping the template to the image, we need to invert it.
    
#     # TODO: Create mask of same size as template
#     th, tw = template.shape[:2]
#     mask = np.ones((th, tw), dtype=np.uint8)

#     # TODO: Warp mask by appropriate homography
#     warped_mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

#     # TODO: Warp template by appropriate homography
#     warped_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
#     mask_inv = 1 - warped_mask
#     # TODO: Use mask to combine the warped template and the image
#     composite_img = img.copy()
#     for c in range(img.shape[2]):
#         composite_img[:, :, c] = (img[:, :, c] * mask_inv + warped_template[:, :, c]*warped_mask).astype(np.uint8)
    
#     return composite_img

# def applyH(H, x):
#     num_points = x.shape[0]
#     homogeneous_points = np.hstack((x, np.ones((num_points, 1))))
#     transformed_points = np.dot(H, homogeneous_points.T).T
#     transformed_points = transformed_points[:, :2] / transformed_points[:, 2][:,np.newaxis]
#     return transformed_points


import numpy as np
import cv2

def computeH(x1, x2):
    A = np.zeros((2 * len(x1), 9))
    for i in range(len(x1)):
        x, y = x1[i]
        u, v = x2[i]
        A[2 * i] = [-u, -v, -1, 0, 0, 0, u * x, v * x, x]
        A[2 * i + 1] = [0, 0, 0, -u, -v, -1, u * y, v * y, y]
    _, _, Vt = np.linalg.svd(A)
    H2to1 = Vt[-1].reshape(3, 3)
    return H2to1

def computeH_norm(x1, x2):
    x1_centroid = np.mean(x1, axis=0)
    x2_centroid = np.mean(x2, axis=0)

    x1_shifted = x1 - x1_centroid
    x2_shifted = x2 - x2_centroid

    x1_max_dist = np.max(np.linalg.norm(x1_shifted, axis=1))
    x2_max_dist = np.max(np.linalg.norm(x2_shifted, axis=1))

    x1_scale = np.sqrt(2) / x1_max_dist
    x2_scale = np.sqrt(2) / x2_max_dist

    # Avoid repetitive calculations
    x1_scale *= x1_shifted.shape[0]
    x2_scale *= x2_shifted.shape[0]

    x1_normalized = x1_shifted * x1_scale
    x2_normalized = x2_shifted * x2_scale

    T1 = np.array([[x1_scale, 0, -x1_scale * x1_centroid[0]],
                   [0, x1_scale, -x1_scale * x1_centroid[1]],
                   [0, 0, 1]])

    T2 = np.array([[x2_scale, 0, -x2_scale * x2_centroid[0]],
                   [0, x2_scale, -x2_scale * x2_centroid[1]],
                   [0, 0, 1]])

    H2to1 = computeH(x1_normalized, x2_normalized)

    H2to1 = np.linalg.inv(T1) @ H2to1 @ T2

    return H2to1

def computeH_ransac(locs1, locs2, opts):
    max_iters = opts.max_iters
    inlier_tol = opts.inlier_tol
    bestH2to1 = None
    best_inliers = []

    def compute_inliers(random_indices):
        random_locs1 = locs1[random_indices]
        random_locs2 = locs2[random_indices]
        H2to1 = computeH_norm(random_locs1, random_locs2)
        transform_locs2 = applyH(H2to1, locs2)
        distances = np.linalg.norm(locs1 - transform_locs2, axis=1)
        inliers = distances < inlier_tol
        return H2to1, inliers

    for _ in range(max_iters):
        random_indices = np.random.choice(locs1.shape[0], 4, replace=False)
        H2to1, inliers = compute_inliers(random_indices)
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            bestH2to1 = H2to1

    return bestH2to1, best_inliers

def compositeH(H2to1, template, img):
    th, tw = template.shape[:2]
    mask = np.ones((th, tw), dtype=np.uint8)
    warped_mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    warped_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    mask_inv = 1 - warped_mask
    composite_img = img.copy()
    for c in range(img.shape[2]):
        composite_img[:, :, c] = (img[:, :, c] * mask_inv + warped_template[:, :, c] * warped_mask).astype(np.uint8)
    return composite_img

def applyH(H, x):
    num_points = x.shape[0]
    homogeneous_points = np.hstack((x, np.ones((num_points, 1))))
    transformed_points = np.dot(H, homogeneous_points.T).T
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]
    return transformed_points

