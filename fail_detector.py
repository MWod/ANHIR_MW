import numpy as np
import scipy.ndimage as nd
import cv2
import utils
from skimage import filters
import mind_2d as mind


def detect_cv_failure(source, target, transform, echo=True):
    if np.sum(np.isnan(transform)) != 0:
        score = 0
        failed = True
    else:
        if not np.isfinite(np.linalg.cond(transform)):
            score = 0
            failed = True
        else:
            u_x, u_y = utils.rigid_dot(source, np.linalg.inv(transform))
            transformed_source = utils.warp_image(source, u_x, u_y)
            ret_source, thresholded_source = threshold_calculation(source)
            ret_target, thresholded_target = threshold_calculation(target)
            ret_transformed_source, thresholded_transformed_source = threshold_calculation_with_threshold(transformed_source, ret_source)

            initial_dice = utils.dice(thresholded_source, thresholded_target)
            transformed_dice = utils.dice(thresholded_transformed_source, thresholded_target)
            if echo:
                print("Initial dice: ", initial_dice)
                print("Transformed dice: ", transformed_dice)

            score = transformed_dice
            if transformed_dice > initial_dice and transformed_dice > 0.75:
                failed = False
            else:
                failed = True

    success = not failed
    return score, success

def detect_ng_failure(source, target, transformed_source, echo=True):
    failed = False
    ret_source, thresholded_source = threshold_calculation(source)
    ret_target, thresholded_target = threshold_calculation(target)
    ret_transformed_source, thresholded_transformed_source = threshold_calculation_with_threshold(transformed_source, ret_source)

    initial_dice = utils.dice(thresholded_source, thresholded_target)
    transformed_dice = utils.dice(thresholded_transformed_source, thresholded_target)
    if echo:
        print("Initial dice: ", initial_dice)
        print("Transformed dice: ", transformed_dice)
    score = transformed_dice
    if transformed_dice > initial_dice - 0.01 and transformed_dice > 0.5:
        failed = False
    else:
        failed = True
    success = not failed
    return score, success

def detect_mind_failure(source, target, transformed_source, echo=True):
    params = dict()
    params['radius'] = (2, 2)
    params['sigma'] = (1.0, 1.0)
    source_target_mind = -mind.mind_ssd(source, target, **params)
    transformed_source_target_mind = -mind.mind_ssd(transformed_source, target, **params)
    if echo:
        print("Source/Target MIND: ", source_target_mind)
        print("Transformed Source/Target MIND: ", transformed_source_target_mind)
    return transformed_source_target_mind <= source_target_mind

def threshold_calculation(image):
    ret_i, th_i = cv2.threshold(image, image.mean(), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_size = np.size(image)
    bw_size = th_i.sum() / 255
    if bw_size / image_size > 0.5 or bw_size / image_size < 0.25:
        for i in range(250, 10, -1):
            ret_bw, th_bw = cv2.threshold(image, i, 255, cv2.THRESH_BINARY)
            bw_size = th_bw.sum() / 255
            if bw_size / image_size < 0.5 and bw_size / image_size > 0.25:
                break
    else:
        ret_bw = ret_i
        th_bw = th_i
    return ret_bw, th_bw

def threshold_calculation_with_rotation(image):
    ret_i = filters.threshold_li(image)
    th_i = image > ret_i
    th_i = th_i*255
    image_size = np.size(image)
    bw_size = th_i.sum() / 255
    if bw_size / image_size > 0.9 or bw_size / image_size < 0.20:
        for i in range(250,10,-1):
            ret_bw,th_bw = cv2.threshold(image, i, 255, cv2.THRESH_BINARY)
            bw_size = th_bw.sum() / 255
            if bw_size / image_size < 0.9 and bw_size / image_size > 0.20:
                break
    else:
        ret_bw=ret_i
        th_bw=th_i
    labeled_mask, cc_num = nd.label((th_bw/255).astype(bool))   
    th_bw = (labeled_mask == (np.bincount(labeled_mask.flat)[1:].argmax() + 1))
    th_bw = (th_bw*255).astype(np.uint8)
    th_bw = nd.binary_fill_holes(th_bw)
    return ret_bw, th_bw   

def threshold_calculation_with_threshold(image, threshold):
    ret_bw, th_bw = cv2.threshold(image, int(threshold), 255, cv2.THRESH_BINARY)
    return ret_bw, th_bw

def threshold_calculation_with_threshold_with_rotation(image, threshold):
    ret_bw, th_bw = cv2.threshold(image, int(threshold), 255,cv2.THRESH_BINARY)
    labeled_mask, cc_num = nd.label((th_bw / 255).astype(bool))   
    th_bw = (labeled_mask == (np.bincount(labeled_mask.flat)[1:].argmax() + 1))
    th_bw = (th_bw*255).astype(np.uint8)
    th_bw = nd.binary_fill_holes(th_bw)
    return ret_bw, th_bw
