import time
import numpy as np
import matplotlib.pyplot as plt

import preprocessing as pp
import initial_alignment as ia
import fail_detector as fd
import nonrigid_registration as nr
import utils


def anhir_method(source, target, echo=True):
    ##### Step 0 - Parameters, Smoothing and Initial Resampling #####

    b_time_total = time.time()
    params = dict()
    params["echo"] = echo
    params["initial_alignment_size"] = 2048
    params["centroid_rotation_size"] = 512
    params["nonrigid_registration_size"] = 2048
    params["gaussian_divider"] = 1.24
    params['nr_method'] = "dm" # pd or dm

    nr_params = dict()
    nr_params['echo'] = echo
    nr_params['global_min_size'] = 64
    nr_params['global_max_size'] = 512
    nr_params['local_min_size'] = 64
    nr_params['local_max_size'] = 768
    nr_params['global_iterations'] = 100
    nr_params['inner_iterations'] = 15
    nr_params['outer_iterations'] = 5
    nr_params['L_smooth'] = 1e7
    nr_params['L_sigma'] = 1
    nr_params['R_sigma'] = 1
    nr_params['M_sigma'] = 2
    nr_params['x_box'] = 19
    nr_params['y_box'] = 19

    nr_params['spacing'] = (1.0, 1.0)
    nr_params['update_mode'] = "composition"
    nr_params['gradient_mode'] = "symmetric"
    nr_params['diffusion_sigma'] = (2.0, 2.0)
    nr_params['fluid_sigma'] = (0.5, 0.5)
    nr_params['mind_sigma'] = (1.0, 1.0)
    nr_params['mind_radius'] = (2, 2)
    nr_params['early_stop'] = 10

    return_dict = dict()

    initial_resample_ratio = utils.calculate_resample_size(source, target, max(params["initial_alignment_size"], params["nonrigid_registration_size"]))

    source = utils.gaussian_filter(source, initial_resample_ratio / params["gaussian_divider"])
    target = utils.gaussian_filter(target, initial_resample_ratio / params["gaussian_divider"])

    if echo:
        print()
        print("Registration start.")
        print()
        print("Source shape: ", source.shape)
        print("Target shape: ", target.shape)


    ##### Step 1 - Preprocessing #####


    if echo:
        print()
        print("Preprocessing start.")
        print()

    b_time_r = time.time()
    p_source, p_target = utils.resample_both(source, target, initial_resample_ratio)
    e_time_r = time.time()
    tt_source = p_source.copy()

    if echo:
        print("Initially resampled source shape: ", p_source.shape)
        print("Initially resampled target shape: ", p_target.shape)
        print("Time for initial resampling: ", e_time_r - b_time_r, " seconds.")

    b_time_p = time.time()
    p_source, p_target, t_source, t_target, source_shift, target_shift = pp.preprocess(p_source, p_target, echo)
    e_time_p = time.time()

    return_dict["preprocessing_time"] = e_time_p - b_time_p

    if echo:
        print("Source shift: ", source_shift)
        print("Target shift: ", target_shift)
        print("Preprocessed source shape: ", p_source.shape)
        print("Preprocessed target shape: ", p_target.shape)
        print("Time for preprocessing: ", e_time_p - b_time_p, " seconds.")
        print()
        print("Preprocessing end.")
        print()


    ##### Step 2 - Initial Alignment #####


    b_ia_time = time.time()

    if echo:
        print("Initial alignment start.")
        print()

    ia_resample_ratio = params["nonrigid_registration_size"] / params["initial_alignment_size"]
    to_cv_source, to_cv_target = utils.resample_both(p_source, p_target, ia_resample_ratio)

    cv_failed = False
    ct_failed = False
    ia_failed = False
    i_u_x, i_u_y, initial_transform, cv_failed = ia.cv_initial_alignment(to_cv_source, to_cv_target, echo)

    if cv_failed:
        if echo:
            print("CV failed.")
            print()
            print("CT start.")
            print()

        ia_resample_ratio = params["nonrigid_registration_size"] / params["centroid_rotation_size"]
        to_ct_source, to_ct_target = utils.resample_both(p_source, p_target, ia_resample_ratio)

        i_u_x, i_u_y, initial_transform, ct_failed = ia.ct_initial_alignment(to_ct_source, to_ct_target, echo)
        if ct_failed:
            if echo:
                print()
                print("CT failed.")
                print("Initial alignment failed.")
            ia_failed = True

    if ia_failed:
        i_u_x, i_u_y = np.zeros(p_source.shape), np.zeros(p_target.shape)
    else:
        y_size, x_size = np.shape(p_source)
        i_u_x, i_u_y = utils.resample_displacement_field(i_u_x, i_u_y, x_size, y_size)

    e_ia_time = time.time()

    return_dict["cv_failed"] = cv_failed
    return_dict["ct_failed"] = ct_failed
    return_dict["ia_failed"] = ia_failed
    return_dict["initial_alignment_time"] = e_ia_time - b_ia_time

    if echo:
        print()
        print("Elapsed time for initial alignment: ", e_ia_time - b_ia_time, " seconds.")
        print("Initial alignment end.")
        print()


    ia_source = utils.warp_image(p_source, i_u_x, i_u_y)
    u_x_g, u_y_g = nr.partial_data_registration_global(ia_source, p_target, nr_params)
    u_x_g, u_y_g = utils.compose_vector_fields(i_u_x, i_u_y, u_x_g, u_y_g)
    ng_source = utils.warp_image(p_source, u_x_g, u_y_g)

    success = fd.detect_mind_failure(ia_source, p_target, ng_source, echo)
    if not success:
        u_x_g, u_y_g = i_u_x, i_u_y
        ng_source = ia_source

    return_dict["ng_failed"] = not success

    ##### Step 3 - Nonrigid Registration #####

    b_nr_time = time.time()

    if echo:
        print("Nonrigid registration start.")
        print()


    if params['nr_method'] == "dm":
        u_x_nr, u_y_nr = nr.dm(ng_source, p_target, nr_params)
        u_x_nr, u_y_nr = utils.compose_vector_fields(u_x_g, u_y_g, u_x_nr, u_y_nr)
        nr_source = utils.warp_image(p_source, u_x_nr, u_y_nr)
    elif params['nr_method'] == "pd":
        u_x_nr, u_y_nr = nr.partial_data_registration_local(ng_source, p_target, nr_params)
        u_x_nr, u_y_nr = utils.compose_vector_fields(u_x_g, u_y_g, u_x_nr, u_y_nr)
        nr_source = utils.warp_image(p_source, u_x_nr, u_y_nr)

    e_nr_time = time.time()

    
    return_dict["nonrigid_registration_time"] = e_nr_time - b_nr_time

    if echo:
        print()
        print("Elapsed time for nonrigid registration: ", e_nr_time - b_nr_time, " seconds.")
        print("Nonrigid registration end.")
        print()


    ##### Step 4 - Warping function creation #####


    def warp_original_landmarks(source_landmarks):
        source_landmarks = source_landmarks / initial_resample_ratio
        source_landmarks = utils.pad_landmarks(source_landmarks, target_shift[0], target_shift[2])
        source_landmarks = utils.transform_landmarks(source_landmarks, u_x_nr, u_y_nr)
        source_l_x, source_r_x, source_l_y, source_r_y = source_shift
        target_l_x, target_r_x, target_l_y, target_r_y = target_shift
        source_landmarks[:, 0] = source_landmarks[:, 0] - source_l_x
        source_landmarks[:, 1] = source_landmarks[:, 1] - source_l_y
        out_y_size, out_x_size = np.shape(source)
        in_y_size, in_x_size = np.shape(tt_source)
        source_landmarks[:, 0] = source_landmarks[:, 0] * out_x_size / in_x_size
        source_landmarks[:, 1] = source_landmarks[:, 1] * out_y_size / in_y_size
        return source_landmarks

    def warp_resampled_landmarks(source_landmarks, target_landmarks):
        source_landmarks = source_landmarks / initial_resample_ratio
        target_landmarks = target_landmarks / initial_resample_ratio
        source_landmarks = utils.pad_landmarks(source_landmarks, target_shift[0], target_shift[2])
        target_landmarks = utils.pad_landmarks(target_landmarks, source_shift[0], source_shift[2])
        transformed_source_landmarks = utils.transform_landmarks(source_landmarks, u_x_nr, u_y_nr)
        return source_landmarks, transformed_source_landmarks, target_landmarks


    e_time_total = time.time()
    return_dict["total_time"] = e_time_total - b_time_total
    if echo:
        print("Total registration time: ", e_time_total - b_time_total, " seconds.")
        print("End of registration.")
        print()


    return p_source, p_target, ia_source, ng_source, nr_source, i_u_x, i_u_y, u_x_nr, u_y_nr, warp_resampled_landmarks, warp_original_landmarks, return_dict

