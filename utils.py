import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as nd
import SimpleITK as sitk
from skimage import color
import csv


def resample(image, output_x_size, output_y_size):
    y_size, x_size = np.shape(image)
    out_grid_x, out_grid_y = np.meshgrid(np.arange(output_x_size), np.arange(output_y_size))
    out_grid_x = out_grid_x * x_size / output_x_size
    out_grid_y = out_grid_y * y_size / output_y_size
    image = nd.map_coordinates(image, [out_grid_y, out_grid_x], order=3, cval=0.0)
    return image

def resample_both(source, target, resample_ratio):
    s_y_size, s_x_size = source.shape
    t_y_size, t_x_size = target.shape
    source = resample(source, int(s_x_size/resample_ratio), int(s_y_size/resample_ratio))
    target = resample(target, int(t_x_size/resample_ratio), int(t_y_size/resample_ratio))
    return source, target

def resample_displacement_field(u_x, u_y, output_x_size, output_y_size):
    y_size, x_size = np.shape(u_x)
    u_x = resample(u_x, output_x_size, output_y_size)
    u_y = resample(u_y, output_x_size, output_y_size)
    u_x = u_x * output_x_size/x_size
    u_y = u_y * output_y_size/y_size
    return u_x, u_y

def warp_image(image, u_x, u_y):
    y_size, x_size = image.shape
    grid_x, grid_y = np.meshgrid(np.arange(x_size), np.arange(y_size))
    return nd.map_coordinates(image, [grid_y + u_y, grid_x + u_x], order=3, cval=0.0)

def rigid_dot(image, matrix):
    y_size, x_size = np.shape(image)
    x_grid, y_grid = np.meshgrid(np.arange(x_size), np.arange(y_size))
    points = np.vstack((x_grid.ravel(), y_grid.ravel(), np.ones(np.shape(image)).ravel()))
    transformed_points = matrix @ points
    u_x = np.reshape(transformed_points[0, :], (y_size, x_size)) - x_grid
    u_y = np.reshape(transformed_points[1, :], (y_size, x_size)) - y_grid
    return u_x, u_y

def load_image(path):
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)
    image = color.rgb2gray(image)
    return image

def load_landmarks(path):
    landmarks = pd.read_csv(path).ix[:, 1:].values.astype(np.float)
    return landmarks

def save_landmarks(path, landmarks):
    df = pd.DataFrame(landmarks, columns=['X', 'Y'])
    df.index = np.arange(1, len(df) + 1)
    df.to_csv(path)

def pad_landmarks(landmarks, x, y):
    landmarks[:, 0] += x
    landmarks[:, 1] += y
    return landmarks

def plot_landmarks(landmarks, marker_type, colors=None):
    landmarks_length = len(landmarks)
    if colors is None:
        colors = np.random.uniform(0, 1, (3, landmarks_length))
    for i in range(landmarks_length):
        plt.plot(landmarks[i, 0], landmarks[i, 1], marker_type, color=colors[:, i])
    return colors

def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def to_image(array):
    return sitk.GetImageFromArray((255*array).astype(np.uint8))

def calculate_resample_size(source, target, output_max_size):
    target_y_size, target_x_size = np.shape(target)[0:2]
    source_y_size, source_x_size = np.shape(source)[0:2]

    max_y_size = max(source_y_size, target_y_size)
    max_x_size = max(source_x_size, target_x_size)

    max_dim = max(max_y_size, max_x_size)
    rescale_ratio = max_dim/output_max_size
    return rescale_ratio

def compose_vector_fields(u_x, u_y, v_x, v_y):
    y_size, x_size = np.shape(u_x)
    grid_x, grid_y = np.meshgrid(np.arange(x_size), np.arange(y_size))
    added_y = grid_y + v_y
    added_x = grid_x + v_x
    t_x = nd.map_coordinates(grid_x + u_x, [added_y, added_x], mode='constant', cval=0.0)
    t_y = nd.map_coordinates(grid_y + u_y, [added_y, added_x], mode='constant', cval=0.0)
    n_x, n_y = t_x - grid_x, t_y - grid_y
    indexes_x = np.logical_or(added_x >= x_size - 1, added_x <= 0)
    indexes_y = np.logical_or(added_y >= y_size - 1, added_y <= 0)
    indexes = np.logical_or(indexes_x, indexes_y)
    n_x[indexes] = 0.0
    n_y[indexes] = 0.0
    return n_x, n_y

def gaussian_filter(image, sigma):
    return nd.gaussian_filter(image, sigma)

def round_up_to_odd(value):
    return int(np.ceil(value) // 2 * 2 + 1)

def dice(image_1, image_2):
    image_1 = image_1.astype(np.bool)
    image_2 = image_2.astype(np.bool)
    return 2 * np.logical_and(image_1, image_2).sum() / (image_1.sum() + image_2.sum())

def transform_landmarks(landmarks, u_x, u_y):
    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    ux = nd.map_coordinates(u_x, [landmarks_y, landmarks_x], mode='nearest')
    uy = nd.map_coordinates(u_y, [landmarks_y, landmarks_x], mode='nearest')
    new_landmarks = np.stack((landmarks_x + ux, landmarks_y + uy), axis=1)
    return new_landmarks

def tre(landmarks_1, landmarks_2):
    tre = np.sqrt(np.square(landmarks_1[:, 0] - landmarks_2[:, 0]) + np.square(landmarks_1[:, 1] - landmarks_2[:, 1]))
    return tre

def rtre(landmarks_1, landmarks_2, x_size, y_size):
    return tre(landmarks_1, landmarks_2) / np.sqrt(x_size*x_size + y_size*y_size)

def print_rtre(source_landmarks, target_landmarks, x_size, y_size):
    calculated_tre = rtre(source_landmarks, target_landmarks, x_size, y_size)
    mean = np.mean(calculated_tre) * 100
    median = np.median(calculated_tre) * 100
    mmax = np.max(calculated_tre) * 100
    mmin = np.min(calculated_tre) * 100
    print("TRE mean [%]: ", mean)
    print("TRE median [%]: ", median)
    print("TRE max [%]: ", mmax)
    print("TRE min [%]: ", mmin)
    return mean, median, mmax, mmin


