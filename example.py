import os
import numpy as np
import matplotlib.pyplot as plt

import anhir_method as am
import utils


def main():
    source_path = None # Source path
    target_path = None # Target path

    source_landmarks_path = None # Source landmarks path
    target_landmarks_path = None # Target landmarks path

    source_landmarks = utils.load_landmarks(source_landmarks_path)
    target_landmarks = utils.load_landmarks(target_landmarks_path)
    source = utils.load_image(source_path)
    target = utils.load_image(target_path)

    p_source, p_target, ia_source, ng_source, nr_source, i_u_x, i_u_y, u_x_nr, u_y_nr, warp_resampled_landmarks, warp_original_landmarks, return_dict = am.anhir_method(target, source)

    transformed_source_landmarks = warp_original_landmarks(source_landmarks)

    resampled_source_landmarks, transformed_resampled_source_landmarks, resampled_target_landmarks = warp_resampled_landmarks(source_landmarks, target_landmarks)

    y_size, x_size = np.shape(target)
    print("Initial original rTRE: ")
    utils.print_rtre(source_landmarks, target_landmarks, x_size, y_size)
    print("Transformed original rTRE: ")
    utils.print_rtre(transformed_source_landmarks, target_landmarks, x_size, y_size)

    y_size, x_size = np.shape(p_target)
    print("Initial resampled rTRE: ")
    utils.print_rtre(resampled_source_landmarks, resampled_target_landmarks, x_size, y_size)
    print("Transformed resampled rTRE: ")
    utils.print_rtre(transformed_resampled_source_landmarks, resampled_target_landmarks, x_size, y_size)

    print(return_dict)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(source, cmap='gray')
    colors = utils.plot_landmarks(source_landmarks, "*")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(target, cmap='gray')
    utils.plot_landmarks(target_landmarks, "*", colors)
    utils.plot_landmarks(transformed_source_landmarks, ".", colors)
    plt.axis('off')

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(p_target, cmap='gray')
    colors = utils.plot_landmarks(resampled_source_landmarks, "*")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(p_source, cmap='gray')
    utils.plot_landmarks(resampled_target_landmarks, "*", colors)
    utils.plot_landmarks(transformed_resampled_source_landmarks, ".", colors)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(nr_source, cmap='gray')
    plt.axis('off')


    plt.show()




if __name__ == "__main__":
    main()