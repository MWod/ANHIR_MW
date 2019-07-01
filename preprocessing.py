import os
import numpy as np
import time
from skimage import filters
from skimage import morphology
import utils


def preprocess(source, target, echo=True):    
    def image_entropy(image):
        return filters.rank.entropy(image, morphology.disk(3))

    def histogram_correction(source, target):
        oldshape = source.shape
        source = source.ravel()
        target = target.ravel()

        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(target, return_counts=True)

        sq = np.cumsum(s_counts).astype(np.float64)
        sq /= sq[-1]
        tq = np.cumsum(t_counts).astype(np.float64)
        tq /= tq[-1]
        interp_t_values = np.interp(sq, tq, t_values)
        return interp_t_values[bin_idx].reshape(oldshape)

    def pad_image(image, x_size, y_size):
        image_y_shape, image_x_shape = np.shape(image)
        image_l_x, image_r_x = int(np.floor((x_size - image_x_shape)/2)), int(np.ceil((x_size - image_x_shape)/2))
        image_l_y, image_r_y = int(np.floor((y_size - image_y_shape)/2)), int(np.ceil((y_size - image_y_shape)/2))
        image = np.pad(image, [(image_l_y, image_r_y), (image_l_x, image_r_x)], mode='constant')
        return image, image_l_x, image_r_x, image_l_y, image_r_y

    source = utils.normalize(source)
    target = utils.normalize(target)

    t_source = source.copy()
    t_target = target.copy()

    b_time_entropy = time.time()
    s_y_size, s_x_size = source.shape
    t_y_size, t_x_size = target.shape
    e_resample_ratio = np.max(np.array([s_y_size, s_x_size, t_y_size, t_x_size])) / 512
    e_source, e_target = utils.resample_both(source, target, e_resample_ratio)
    e_source, e_target = utils.normalize(e_source), utils.normalize(e_target)
    source_entropy = image_entropy(e_source)
    target_entropy = image_entropy(e_target)
    e_time_entropy = time.time()

    if echo:
        print("Source entropy: ", np.mean(source_entropy))
        print("Target entropy: ", np.mean(target_entropy))

    if echo:
        print("Time for entropy calculation: ", e_time_entropy - b_time_entropy, " seconds.")

    b_time_hist = time.time()
    if np.mean(target_entropy) > np.mean(source_entropy):
        source = histogram_correction(source, target)
    else:
        target = histogram_correction(target, source)
    e_time_hist = time.time()

    if echo:
        print("Time for histogram correction: ", e_time_hist - b_time_hist, " seconds.")

    source = utils.normalize(source)
    target = utils.normalize(target)

    source = 1 - source
    target = 1 - target

    t_source = 1 - t_source
    t_target = 1 - t_target

    x_size = max(source.shape[1], target.shape[1])
    y_size = max(source.shape[0], target.shape[0])

    source, source_l_x, source_r_x, source_l_y, source_r_y = pad_image(source, x_size, y_size)
    target, target_l_x, target_r_x, target_l_y, target_r_y = pad_image(target, x_size, y_size)

    t_source, _, _, _, _ = pad_image(t_source, x_size, y_size)
    t_target, _, _, _, _ = pad_image(t_target, x_size, y_size)

    return source, target, t_source, t_target, (source_l_x, source_r_x, source_l_y, source_r_y), (target_l_x, target_r_x, target_l_y, target_r_y)