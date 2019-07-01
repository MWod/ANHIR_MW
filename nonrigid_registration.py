import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as nd
from scipy import signal

import utils
import demons_mind_2d_cpp as dmc


def partial_data_registration(source, target, params):
    u_x_g, u_y_g = partial_data_registration_global(source, target, params)
    source = utils.warp_image(source, u_x_g, u_y_g)
    u_x_l, u_y_l = partial_data_registration_local(source, target, params)
    u_x_t, u_y_t = utils.compose_vector_fields(u_x_g, u_y_g, u_x_l, u_y_l)
    return u_x_g, u_y_g, u_x_t, u_y_t

def dm(source, target, params):
    echo = params['echo']
    y_size, x_size = np.shape(source)
    spacing = params['spacing']
    update_mode = params['update_mode']
    gradient_mode = params['gradient_mode']
    diffusion_sigma = params['diffusion_sigma']
    fluid_sigma = params['fluid_sigma']
    mind_sigma = params['mind_sigma']
    mind_radius = params['mind_radius']
    early_stop = params['early_stop']
    resolutions, iters = calculate_resolutions_and_iters(y_size, x_size)
    max_iterations = iters[0]

    u_x, u_y = dmc.demons_mind_registration(source, target, spacing, update_mode=update_mode, gradient_mode=gradient_mode,
        resolutions=resolutions, diffusion_sigma=diffusion_sigma, fluid_sigma=fluid_sigma, mind_sigma=mind_sigma, mind_radius=mind_radius,
        max_iterations=max_iterations, return_best=True, iterations=iters, early_stop=early_stop,
        echo=echo)

    return u_x, u_y

def calculate_resolutions_and_iters(y_size, x_size):
    minimum_size = 256
    iters_at_minimum = 500
    current_size = min(y_size, x_size)
    resolutions = int(np.floor(np.log2(current_size) - np.log(minimum_size)))
    iters = list()
    iters.append(iters_at_minimum)
    for i in range(1, int(resolutions)):
        iters.append(int(iters[i-1] / 2))
    return resolutions, iters

def partial_data_registration_local(source, target, params):
    echo = params['echo']
    o_y_size, o_x_size = source.shape
    source, target = initial_resample(source, target, params['local_max_size'], params)
    y_size, x_size = source.shape
    image_min_size = min(y_size, x_size)
    min_size = params['local_min_size']
    min_ratio = image_min_size / min_size
    levels = int(np.log2(np.floor(min_ratio))) + 1
    sources, targets = build_pyramids(source, target, levels)
    if echo:
        print("Local registration started.")
    for i in range(levels):
        if echo:
            print()
            print("Current level: %f/%f" % (i+1, levels))
            print()
        current_source = sources[i]
        current_target = targets[i]
        if i == 0:
            u_x, u_y = np.zeros(current_source.shape), np.zeros(current_source.shape)
        if i != 0: 
            current_source = utils.warp_image(current_source, u_x, u_y)
        t_u_x, t_u_y = single_resolution_local(current_source, current_target, i, params)
        u_x, u_y = utils.compose_vector_fields(u_x, u_y, t_u_x, t_u_y)
        if i != levels - 1:
            ys, xs = sources[i+1].shape
            u_x, u_y = utils.resample_displacement_field(u_x, u_y, xs, ys)
    u_x, u_y = utils.resample_displacement_field(u_x, u_y, o_x_size, o_y_size)
    return u_x, u_y

def partial_data_registration_global(source, target, params):
    echo = params['echo']
    o_y_size, o_x_size = source.shape
    source, target = initial_resample(source, target, params['global_max_size'], params)
    y_size, x_size = source.shape
    image_min_size = min(y_size, x_size)
    min_size = params['global_min_size']
    min_ratio = image_min_size / min_size
    levels = int(np.log2(np.floor(min_ratio))) + 1
    sources, targets = build_pyramids(source, target, levels)
    if echo:
        print("Global registration started.")
    for i in range(levels):
        if echo:
            print()
            print("Current level: %d/%d" % (i+1, levels))
            print()
        current_source = sources[i]
        current_target = targets[i]
        if i == 0:
            u_x, u_y = np.zeros(current_source.shape), np.zeros(current_source.shape)   
        current_source = utils.warp_image(current_source, u_x, u_y)
        t_u_x, t_u_y = single_resolution_global(current_source, current_target, i, params)
        u_x, u_y = utils.compose_vector_fields(u_x, u_y, t_u_x, t_u_y)
        if i != levels - 1:
            ys, xs = sources[i+1].shape
            u_x, u_y = utils.resample_displacement_field(u_x, u_y, xs, ys)
    u_x, u_y = utils.resample_displacement_field(u_x, u_y, o_x_size, o_y_size)
    return u_x, u_y

def build_pyramids(source, target, levels):
    sources = [None] * levels
    targets = [None] * levels
    sources[-1] = source
    targets[-1] = target
    source_sm = nd.gaussian_filter(source, 3)
    target_sm = nd.gaussian_filter(target, 3)
    for i in range(levels - 1):
        sources[i], targets[i] = utils.resample_both(source_sm, target_sm, 2**(levels-i-1))
    return sources, targets

def initial_resample(source, target, max_size, params):
    y_size, x_size = source.shape
    image_min_size = min(y_size, x_size)
    if image_min_size > max_size:
        ratio = image_min_size / max_size
        source, target = utils.resample_both(source, target, ratio)
    return source, target

def single_resolution_global(source, target, level, params):
    echo = params['echo']
    y_size, x_size = source.shape
    u_x = np.zeros(source.shape)
    u_y = np.zeros(target.shape)
    grid_x, grid_y = np.meshgrid(np.arange(x_size), np.arange(y_size))

    t_grid_x = grid_x - np.max(grid_x) / 2
    t_grid_y = grid_y - np.max(grid_y) / 2
    t_grid_y = -t_grid_y

    iterations = params['global_iterations']
    transformed_source = source.copy()
    for iteration in range(int(iterations / 2**(level))):
        if echo:
            print("Current global iteration: ", iteration)
        transform = global_transform(transformed_source, target, t_grid_x, t_grid_y, params)
        t_u_x, t_u_y = transform_to_displacement_field(transform, t_grid_x, t_grid_y)
        u_x, u_y = utils.compose_vector_fields(u_x, u_y, t_u_x, t_u_y)
        transformed_source = utils.warp_image(source, u_x, u_y)
    return u_x, u_y

def single_resolution_local(source, target, level, params):
    echo = params['echo']

    y_size, x_size = source.shape
    u_x = np.zeros(source.shape)
    u_y = np.zeros(target.shape)

    grid_x, grid_y = np.meshgrid(np.arange(x_size), np.arange(y_size))
    t_grid_x = grid_x - np.max(grid_x) / 2
    t_grid_y = grid_y - np.max(grid_y) / 2
    t_grid_y = -t_grid_y

    outer_iterations = params['outer_iterations']
    transformed_source = source.copy()
    for outer_iteration in range(outer_iterations):
        if echo:
            print("Current outer iteration: ", outer_iteration)
        grad_x, grad_y = gradient_both(transformed_source, target)
        diff = transformed_source - target
        transform, offrange = local_transform(transformed_source, target, diff, grad_x, grad_y, t_grid_x, t_grid_y, params)
        if echo:
            print("Initial transform calculated.")
        transform = transform_smoothing(transformed_source, target, transform, diff, grad_x, grad_y, t_grid_x, t_grid_y, offrange, params)
        if echo:
            print("Smoothing completed..")
        t_u_x, t_u_y = transform_to_displacement_field(transform, t_grid_x, t_grid_y)
        u_x, u_y = utils.compose_vector_fields(u_x, u_y, t_u_x, t_u_y)
        transformed_source = utils.warp_image(source, u_x, u_y)
    return u_x, u_y

def global_transform(source, target, grid_x, grid_y, params):
    grad_x, grad_y = gradient_both(source, target) 
    diff = source - target
    transform, _ = transform_search(source, target,
        diff, grad_x, grad_y,
        grid_x, grid_y, params)
    return transform.ravel()

def local_transform(source, target, diff, grad_x, grad_y, grid_x, grid_y, params):
    y_size, x_size = source.shape
    x_box, y_box = params['x_box'], params['y_box']
    x_step, y_step = x_box // 2, y_box // 2
    initial_local_transform = np.array([1, 0, 0, 1, 0, 0, 1, 0]).reshape(1, 1, 8)
    transform = np.repeat(initial_local_transform, y_size, axis=0)
    transform = np.repeat(transform, x_size, axis=1).astype(np.float32)
    xx = grid_x*grad_x
    xy = grid_x*grad_y
    yx = grid_y*grad_x
    yy = grid_y*grad_y
    x_indices = np.arange(0, x_box)
    x_indices = np.tile(x_indices, x_size - 2*x_step).reshape(-1, x_box).T
    x_indices = x_indices + np.arange(x_indices.shape[1])
    ones = np.ones((y_box, x_box, x_indices.shape[1]))
    offrange = np.zeros(source.shape)

    for j in range(y_step, y_size - y_step):
        b_y, e_y = j - y_step, j + y_step + 1
        source_patches = source[b_y:e_y, x_indices]
        target_patches = target[b_y:e_y, x_indices]
        diff_patches = diff[b_y:e_y, x_indices]
        grad_x_patches = grad_x[b_y:e_y, x_indices]
        grad_y_patches = grad_y[b_y:e_y, x_indices]
        xx_patches = xx[b_y:e_y, x_indices]
        yx_patches = yx[b_y:e_y, x_indices]
        xy_patches = xy[b_y:e_y, x_indices]
        yy_patches = yy[b_y:e_y, x_indices]
        k_vectors = (diff_patches - source_patches + xx_patches + yy_patches)
        c_vectors = np.stack((
            xx_patches,
            yx_patches,
            xy_patches,
            yy_patches,
            grad_x_patches,
            grad_y_patches,
            -source_patches,
            -ones
        ))  
        k_vectors = k_vectors.swapaxes(0, 2).swapaxes(1, 2)
        c_vectors = c_vectors.swapaxes(0, 3).swapaxes(1, 3).swapaxes(2, 3)
        k_vectors = k_vectors.reshape(x_indices.shape[1], -1)
        c_vectors = c_vectors.reshape(x_indices.shape[1], 8, -1)
        P = c_vectors @ c_vectors.swapaxes(1, 2)
        K = c_vectors @ k_vectors.reshape(k_vectors.shape + (1,))

        result = np.zeros((x_size - 2*x_step, 8))
        indices = np.linalg.cond(P) < 1e8
        result[indices] = (np.linalg.inv(P[indices, :, :]) @ K[indices])[:, :, 0]
        result[np.logical_not(indices)] = np.array([1, 0, 0, 1, 0, 0, 1, 0])
        offrange[j, np.arange(x_step, x_size - x_step)] = indices.astype(np.int32)
        transform[j, np.arange(x_step, x_size - x_step), :] = result
    return transform, offrange

def transform_search(source, target, diff, grad_x, grad_y, grid_x, grid_y, params):
    k_vector, c_vector = calculate_transform_vectors(source, target, diff, grad_x, grad_y, grid_x, grid_y, params)
    P = c_vector @ c_vector.T
    K = c_vector @ k_vector.T
    if np.linalg.cond(P) < 1e8:
        transform = (np.linalg.inv(P) @ K).T
        offrange = 1
    else:
        transform = np.array([1, 0, 0, 1, 0, 0, 1, 0])
        offrange = 0
    return transform, offrange

def calculate_transform_vectors(source, target, diff, grad_x, grad_y, grid_x, grid_y, params):
    grad_x, grad_y = grad_x.ravel(), grad_y.ravel()
    grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
    diff = diff.ravel()
    ss = source.ravel()
    ones = np.ones(source.shape).ravel()
    xx = grid_x*grad_x
    xy = grid_x*grad_y
    yx = grid_y*grad_x
    yy = grid_y*grad_y

    k_vector = (diff - ss + xx + yy).reshape(1, -1)
    c_vector = np.stack((
        xx,
        yx,
        xy,
        yy,
        grad_x,
        grad_y,
        -ss,
        -ones
        ))
    return k_vector, c_vector

def transform_smoothing(source, target, transform, diff, grad_x, grad_y, grid_x, grid_y, offrange, params):
    echo = params['echo']
    inner_iterations = params['inner_iterations']
    L_smooth = params['L_smooth']
    L_sigma = params['L_sigma']
    R_sigma = params['R_sigma']
    M_sigma = params['M_sigma']

    y_size, x_size, _ = transform.shape
    k_vector, c_vector = calculate_transform_vectors(source, target, diff, grad_x, grad_y, grid_x, grid_y, params)

    k_vector = k_vector.swapaxes(0, 1)
    c_vector = c_vector.swapaxes(0, 1).reshape(-1, 8, 1)

    p_matrix = c_vector @ c_vector.swapaxes(1, 2)
    k_matrix = c_vector[:, :, 0] * k_vector
    l_matrix = (np.eye(8) * L_smooth).reshape(1, 8, 8)
    l_matrix = np.repeat(l_matrix, source.size, axis=0)

    r_1 = source - target
    r_1 = r_1**2
    r_1 = nd.gaussian_filter(r_1, sigma=R_sigma)
    r_1 = r_1.ravel()
    r_2 = L_sigma / 10

    error_1 = np.exp(-r_1/L_sigma)
    error_2 = np.exp(-r_2/L_sigma)

    w = error_1 / (error_1 + error_2)
    w = w / np.max(w)
    w = w.reshape(-1, 1, 1)
    w = np.repeat(w, 8, axis=1)

    m1 = (c_vector * w) @ c_vector.swapaxes(1, 2)
    m2 = c_vector * w * k_vector.reshape(-1, 1, 1)
    L = l_matrix
    m1_inv = np.linalg.inv(m1 + L)

    offrange_temp = offrange.copy()
    transform_avg = transform.copy()
    for inner_iteration in range(inner_iterations):
        offrange_smooth = nd.gaussian_filter(offrange_temp, 1)
        offrange_indices = offrange_smooth == 0
        offrange_temp = np.ones(source.shape)
        offrange_temp[offrange_indices] = 0
        transform_avg = transform_avg.reshape(y_size, x_size, 8)
        for z in range(8):
            if z != 6 or z != 7:
                transform_avg[:, :, z] = nd.gaussian_filter(transform_avg[:, :, z], sigma=M_sigma)
            else:
                transform_avg[:, :, z] = nd.gaussian_filter(transform_avg[:, :, z], sigma=M_sigma / 20)
        transform_avg = transform_avg.reshape(-1, 8, 1)
        first_term = m1_inv
        second_term = m2 + (L @ transform_avg)
        transform_avg = (first_term @ second_term.reshape(-1, 8, 1)).reshape(-1, 8)
        transform_avg[offrange_indices.ravel(), :] = np.array([1, 0, 0, 1, 0, 0, 1, 0])

    transform_avg = transform_avg.reshape(y_size, x_size, 8)
    transform = transform_avg
    return transform

def gradient(image):
    p = np.array([0.0377, 0.2492, 0.4264, 0.2492, 0.0377]).astype(np.float32)
    d = np.array([0.1096, 0.2767, 0, -0.2767, -0.1096]).astype(np.float32)
    grad_x = -signal.sepfir2d(image, d, p)
    grad_y = signal.sepfir2d(image, p, d)
    return grad_x, grad_y

def gradient_both(source, target):
    image = (source + target) / 2
    return gradient(image)

def transform_to_displacement_field(transform, grid_x, grid_y):
    if transform.ndim == 3:
        u_x = transform[:, :, 0] * grid_x + transform[:, :, 1] * grid_y + transform[:, :, 4] - grid_x
        u_y = transform[:, :, 2] * grid_x + transform[:, :, 3] * grid_y + transform[:, :, 5] - grid_y
        u_y = -u_y
    else:
        u_x = transform[0] * grid_x + transform[1] * grid_y + transform[4] - grid_x
        u_y = transform[2] * grid_x + transform[3] * grid_y + transform[5] - grid_y
        u_y = -u_y
    return u_x, u_y


def example():
    source_path = r"/home/mw/MW_Learning/ANHIR_Results/ia_test/6/source.png"
    target_path = r"/home/mw/MW_Learning/ANHIR_Results/ia_test/6/target_ia.png"


    source = utils.load_image(source_path)
    target = utils.load_image(target_path)

    source, target = utils.resample_both(source, target, 8)
    source = utils.normalize(source).astype(np.float32)
    target = utils.normalize(target).astype(np.float32)

    params = dict()
    params['echo'] = True
    params['global_min_size'] = 64
    params['global_max_size'] = 512
    params['local_min_size'] = 64
    params['local_max_size'] = 1024
    params['global_iterations'] = 100
    params['inner_iterations'] = 15
    params['outer_iterations'] = 5
    params['L_smooth'] = 1e4
    params['L_sigma'] = 1
    params['R_sigma'] = 1
    params['M_sigma'] = 3
    params['x_box'] = 15
    params['y_box'] = 15

    b_t = time.time()
    # u_x_g, u_y_g, u_x, u_y = partial_data_registration(target, source, params)
    u_x_g, u_y_g, u_x, u_y = dm(target, source, params)
    e_t = time.time()
    print("Total time: ", e_t - b_t, " seconds.")
    transformed_target_global = utils.warp_image(target, u_x_g, u_y_g)
    transformed_target = utils.warp_image(target, u_x, u_y)

    num_cols = 4
    num_rows = 2
    plt.figure()
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(source, cmap='gray')
    plt.title("Source")
    plt.axis('off')
    plt.subplot(num_rows, num_cols, 2)
    plt.imshow(target, cmap='gray')
    plt.title("Target")
    plt.axis('off')
    plt.subplot(num_rows, num_cols, 3)
    plt.imshow(transformed_target_global, cmap='gray')
    plt.title("Transformed Target Global")
    plt.axis('off')
    plt.subplot(num_rows, num_cols, 4)
    plt.imshow(transformed_target, cmap='gray')
    plt.title("Transformed Target")
    plt.axis('off')
    plt.subplot(num_rows, num_cols, 5)
    plt.imshow(np.abs(transformed_target_global - source), cmap='gray', vmin=0, vmax=1)
    plt.title("Global diff")
    plt.axis('off')
    plt.subplot(num_rows, num_cols, 6)
    plt.imshow(np.abs(transformed_target - source), cmap='gray', vmin=0, vmax=1)
    plt.title("Global + Local diff")
    plt.axis('off')
    plt.subplot(num_rows, num_cols, 7)
    plt.imshow(np.sqrt(np.square(u_x_g - u_x) + np.square(u_y_g - u_y)), cmap='gray')
    plt.title("DF diff")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    example()

