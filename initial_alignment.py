import os
import time
import numpy as np
import numpy.matlib as matlib
import scipy.ndimage as nd
import cv2
import matplotlib.pyplot as plt


import fail_detector as fd
import utils


def cv_initial_alignment(source, target, echo=True):
    failed = False
    try:
        y_size, x_size = source.shape
        source = (source * 255).astype(np.uint8)
        target = (target * 255).astype(np.uint8)
        max_size = max(y_size, x_size)
        smoothing_size = utils.round_up_to_odd(max_size / 2048 * 31)
        source = cv2.GaussianBlur(source, (smoothing_size, smoothing_size), 0)
        target = cv2.GaussianBlur(target, (smoothing_size, smoothing_size), 0)

        if echo:
            print("SURFN: ")
            print()
        surfn_transform, surfn_score, surfn_failed = calculate_transform(source, target, echo, "surfn")

        if echo:
            print()
            print("SURFE: ")
            print()
        surfe_transform, surfe_score, surfe_failed = calculate_transform(source, target, echo, "surfe")
        
        if echo:
            print()
            print("ORB: ")
            print()
        orb_transform, orb_score, orb_failed = calculate_transform(source, target, echo, "orb")

        if echo:
            print()
            print("SIFT: ")
            print()
        sift_transform, sift_score, sift_failed = calculate_transform(source, target, echo, "sift")

        if echo:
            print("SURFN score:", surfn_score, "SURFN failed: ", surfn_failed)
            print("SURFE score:", surfe_score, "SURFE failed: ", surfe_failed)
            print("ORB score:", orb_score, "ORB failed: ", orb_failed)
            print("SIFT score:", sift_score, "SIFT failed: ", sift_failed)

        scores = np.array([surfn_score, surfe_score, orb_score, sift_score])
        transforms = np.array([surfn_transform, surfe_transform, orb_transform, sift_transform])
        best_id = np.argmax(scores)

        if scores[best_id] == 0:
            failed = True
            transform = np.eye(3)
            u_x, u_y = np.zeros(source.shape), np.zeros(source.shape)
        else:
            failed = False
            transform = transforms[best_id]
            u_x, u_y = utils.rigid_dot(source, np.linalg.inv(transform))
    except:
        failed = True
        transform = np.eye(3)
        u_x, u_y = np.zeros(source.shape), np.zeros(source.shape)

    return u_x, u_y, transform, failed


def ct_initial_alignment(source, target, echo=True):
    y_size, x_size = source.shape
    source = (source * 255).astype(np.uint8)
    target = (target * 255).astype(np.uint8)
    max_size = max(y_size, x_size)
    smoothing_size = utils.round_up_to_odd(max_size / 2048 * 31)
    source = cv2.GaussianBlur(source, (smoothing_size, smoothing_size), 0)
    target = cv2.GaussianBlur(target, (smoothing_size, smoothing_size), 0)

    ret_source, thresholded_source = fd.threshold_calculation_with_rotation(source)
    ret_target, thresholded_target = fd.threshold_calculation_with_rotation(target)

    xs_m = utils.round_up_to_odd(x_size * 20 / 2048)
    ys_m = utils.round_up_to_odd(y_size * 20 / 2048)

    struct = min([xs_m, ys_m])
    thresholded_source = nd.binary_erosion(thresholded_source, structure=np.ones((struct, struct))).astype(np.uint8)
    thresholded_source = nd.binary_dilation(thresholded_source, structure=np.ones((struct, struct))).astype(np.uint8)                            
    thresholded_target = nd.binary_erosion(thresholded_target, structure=np.ones((struct, struct))).astype(np.uint8)
    thresholded_target = nd.binary_dilation(thresholded_target, structure=np.ones((struct, struct))).astype(np.uint8)

    Ms = cv2.moments(thresholded_source)
    Mt = cv2.moments(thresholded_target)

    cXs = Ms["m10"] / Ms["m00"]
    cYs = Ms["m01"] / Ms["m00"]
    cXt = Mt["m10"] / Mt["m00"]
    cYt = Mt["m01"] / Mt["m00"]

    transform_centroid = np.array([
            [1, 0, (cXt-cXs)],
            [0, 1, (cYt-cYs)],
            [0, 0, 1]])
    u_x_t, u_y_t = utils.rigid_dot(source, np.linalg.inv(transform_centroid))
    failed = True
    angle_step = 2
    initial_dice = utils.dice(thresholded_source, thresholded_target)
    if echo:
        print("Initial dice: ", initial_dice)
    best_dice = initial_dice
    for i in range(0, 360, angle_step):
        if echo:
            print("Current angle: ", i)
        rads = i * np.pi/180
        matrix_1 = np.array([
            [1, 0, cXt],
            [0, 1, cYt],
            [0, 0, 1],
        ])
        matrix_i = np.array([
            [np.cos(rads), -np.sin(rads), 0],
            [np.sin(rads), np.cos(rads), 0],
            [0, 0, 1],
        ])
        matrix_2 = np.array([
            [1, 0, -cXt],
            [0, 1, -cYt],
            [0, 0, 1],
        ])

        matrix = matrix_1 @ matrix_i @ matrix_2
        u_x, u_y = utils.rigid_dot(source, np.linalg.inv(matrix))
        transformed_source = utils.warp_image(source, u_x + u_x_t, u_y + u_y_t)

        ret_transformed_source, thresholded_transformed_source = fd.threshold_calculation_with_threshold_with_rotation(transformed_source, ret_source)
        thresholded_transformed_source = nd.binary_erosion(thresholded_transformed_source, structure=np.ones((struct, struct))).astype(np.uint8)
        thresholded_transformed_source = nd.binary_dilation(thresholded_transformed_source, structure=np.ones((struct, struct))).astype(np.uint8)
        current_dice = utils.dice(thresholded_transformed_source, thresholded_target)
        if echo:
            print("Current dice: ", current_dice)

        if (current_dice > best_dice and current_dice > initial_dice + 0.10 and current_dice > 0.85) or (current_dice > 0.95 and current_dice > best_dice):
            failed = False
            best_dice = current_dice
            transform = matrix.copy()
            if echo:
                print("Current best dice: ", best_dice)

    if failed:
        transform = np.eye(3)

    final_transform = transform @ transform_centroid
    if echo:
        print("Calculated transform: ", final_transform)
    if failed:
        final_transform = np.eye(3)
    u_x, u_y = utils.rigid_dot(source, np.linalg.inv(final_transform))
    return u_x, u_y, final_transform, failed


def calculate_transform(source, target, echo=True, descriptor="surfn"):
    b_t = time.time()
    failed = False
    score = 0
    transform = np.eye(3)
    try:
        if descriptor == "surfn":
            source_keypoints, source_descriptors, target_keypoints, target_descriptors = SURF_calculation(source, target, False)
            source_points, target_points = matcher(source_keypoints, target_keypoints, source_descriptors, target_descriptors)
        elif descriptor == "surfe":
            source_keypoints, source_descriptors, target_keypoints, target_descriptors = SURF_calculation(source, target, True)
            source_points, target_points = matcher(source_keypoints, target_keypoints, source_descriptors, target_descriptors)
        elif descriptor == "orb":
            source_keypoints, source_descriptors, target_keypoints, target_descriptors = ORB_calculation(source, target)
            source_points, target_points = matcher(source_keypoints, target_keypoints, source_descriptors, target_descriptors, True)
        elif descriptor == "sift":
            source_keypoints, source_descriptors, target_keypoints, target_descriptors = SIFT_calculation(source, target)
            source_points, target_points = matcher(source_keypoints, target_keypoints, source_descriptors, target_descriptors)
        else:
            raise ValueError("Unsupported descriptor.")
        
        if echo:
            print("Source size: ", len(source_points))
            print("Target size: ", len(target_points))

        transform_affine, transform_rigid, affine_failed, rigid_failed = find_transform(source, target, source_points, target_points)

        if echo:
            print("Calculated Affine: ", transform_affine)
            print("Calculated Rigid: ", transform_rigid)

        if affine_failed and rigid_failed:
            transform = np.eye(3)
            score = 0
            failed = True
        else:
            fd_affine_success = False
            if not affine_failed:
                affine_score, fd_affine_success = fd.detect_cv_failure(source, target, transform_affine, echo)

            fd_rigid_success = False
            if not rigid_failed:
                rigid_score, fd_rigid_success = fd.detect_cv_failure(source, target, transform_rigid, echo)

            if fd_affine_success and fd_rigid_success:
                if affine_score > rigid_score:
                    transform = transform_affine
                    score = affine_score
                    failed = False
                else:
                    transform = transform_rigid
                    score = rigid_score
                    failed = False
            elif fd_affine_success:
                transform = transform_affine
                score = affine_score
                failed = False
            elif fd_rigid_success:
                transform = transform_rigid
                score = rigid_score
                failed = False
            else:
                transform = np.eye(3)
                score = 0
                failed = True
    except:
        transform = np.eye(3)
        score = 0
        failed = True

    if echo:
        if failed:
            print("Failed.")

    e_t = time.time()
    if echo:
        print("Time: ", e_t - b_t, " seconds.")
        print()

    return transform, score, failed

def SURF_calculation(source, target, extended=False):
    hessian_threshold = 400 # Just a magic number
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    surf.setExtended(extended)
    source_keypoints, source_descriptors = surf.detectAndCompute(source, None)
    target_keypoints, target_descriptors = surf.detectAndCompute(target, None)
    return source_keypoints, source_descriptors, target_keypoints, target_descriptors

def ORB_calculation(source, target):
    # Magic numbers below
    y_size, x_size = source.shape
    max_size = max(y_size, x_size)
    num_features = 5000
    scale_factor = 1.3
    num_levels = 8
    edge_threshold = 60
    first_level = 0
    wta_k = 3
    patch_size = utils.round_up_to_odd(35 * max_size / 2048)
    fast_threshold = utils.round_up_to_odd(25 * max_size / 2048)
    orb = cv2.ORB_create(num_features, scale_factor, num_levels,
        edge_threshold, first_level, wta_k, cv2.ORB_HARRIS_SCORE,
        patch_size, fast_threshold)
    source_keypoints, source_descriptors = orb.detectAndCompute(source, None)
    target_keypoints, target_descriptors = orb.detectAndCompute(target, None)
    return source_keypoints, source_descriptors, target_keypoints, target_descriptors

def SIFT_calculation(source, target):
    sift = cv2.xfeatures2d.SIFT_create()
    source_keypoints, source_descriptors = sift.detectAndCompute(source, None)
    target_keypoints, target_descriptors = sift.detectAndCompute(target, None)
    return source_keypoints, source_descriptors, target_keypoints, target_descriptors

def matcher(source_keypoints, target_keypoints, source_descriptors, target_descriptors, orb=False):
    if not orb:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    else:
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm = FLANN_INDEX_LSH,
            table_number = 12,
            key_size = 20,
            multi_probe_level = 2)
    search_params = dict(checks = 600)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(source_descriptors, target_descriptors, k=2)
    good_matches = []
    for m, n in matches:
       if m.distance < 0.7*n.distance:
           good_matches.append(m)
           
    source_points = np.float32([source_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    target_points = np.float32([target_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    return source_points, target_points

def calculate_scale(transform):
    a, b, c, d = transform[0, 0], transform[0, 1], transform[1, 0], transform[1, 1]
    sx = np.sign(a) * np.sqrt(a*a + b*b)
    sy = np.sign(d) * np.sqrt(c*c + d*d)
    return sx, sy

def find_transform(source, target, source_points, target_points):
    y_size, x_size = np.shape(source)
    max_size = max(y_size, x_size)

    threshold = utils.round_up_to_odd(20 * max_size / 2048)
    transform_affine, ransac_affine_failed = ransac_partially_affine(source_points, target_points, 0.99, threshold)
    if ransac_affine_failed:
        threshold = utils.round_up_to_odd(30 * max_size / 2048)
        transform_affine, ransac_affine_failed = ransac_partially_affine(source_points, target_points, 0.90, threshold)

    threshold = utils.round_up_to_odd(20 * max_size / 2048)
    transform_rigid, ransac_rigid_failed = ransac_rigid(source_points, target_points, 0.99, threshold)
    if ransac_rigid_failed:
        threshold = utils.round_up_to_odd(30 * max_size / 2048)
        transform_rigid, ransac_rigid_failed = ransac_rigid(source_points, target_points, 0.90, threshold)

    return transform_affine, transform_rigid, ransac_affine_failed, ransac_rigid_failed

def ransac_partially_affine(source_points, target_points, confidence, threshold):
    try:
        max_iters = 25000
        transform, _ = cv2.estimateAffinePartial2D(source_points, target_points, 0, ransacReprojThreshold = threshold, maxIters = max_iters, confidence = confidence)
        if transform is not None:
            sx, sy = calculate_scale(transform)
            if abs(sx) < 0.85 or abs(sx) > 1.15 or abs(sy) < 0.85 or abs(sy) > 1.15:
                transform = np.eye(3)
                failed = True
            else:
                t_transform = transform
                transform = np.eye(3)
                transform[0:2, 0:3] = t_transform
                failed = False
        else:
            transform = np.eye(3)
            failed = True
    except:
        transform = np.eye(3)
        failed = True

    return transform, failed


def ransac_rigid(source_points, target_points, confidence, threshold):
    try:
        max_iters = 25000
        transform, inliers = cv2.estimateAffinePartial2D(source_points, target_points, 0, ransacReprojThreshold = threshold, maxIters = max_iters, confidence = confidence)
        source_points = np.squeeze(source_points, None)
        target_points = np.squeeze(target_points, None)
        transform = cv2.estimateRigidTransform(
            np.resize(source_points[matlib.repmat(inliers.astype(bool), 1, 2)],
            (np.sum(inliers), 2)),
            np.resize(target_points[matlib.repmat(inliers.astype(bool), 1, 2)],
            (np.sum(inliers), 2)),
            0)
        if transform is not None:
            t_transform = transform
            transform = np.eye(3)
            transform[0:2, 0:3] = t_transform
            failed = False
        else:
            transform = np.eye(3)
            failed = True
    except:
        transform = np.eye(3)
        failed = True
    return transform, failed


