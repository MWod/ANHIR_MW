import os
import shutil
import time

import numpy as np
import SimpleITK as sitk
import pandas as pd

import anhir_method as am
import utils
import json


def run():
    ids_to_run = list(range(0, 481))
    data_path = None # Path to data folder
    csv_file_path = None # Path to the CSV with registration pairs
    results_path = None # Path to the result folder
    output_csv_path = None # Path to the output CSV

    if not os.path.isfile(output_csv_path):
        df = pd.read_csv(csv_file_path)
        del df['Unnamed: 0']
        i_tre = np.empty(len(df))
        i_tre[:] = np.nan
        f_tre = np.empty(len(df))
        f_tre[:] = np.nan
        df['Initial TRE Median'] = i_tre
        df['Final TRE Median'] = f_tre
        df.to_csv(output_csv_path)  

    for current_id in ids_to_run:
        dataframe = pd.read_csv(output_csv_path)
        del dataframe['Unnamed: 0']
        print("Current ID: ",  current_id)
        source_path = dataframe['Source image'][current_id]
        target_path = dataframe['Target image'][current_id]
        source_landmarks_path = dataframe['Source landmarks'][current_id]
        target_landmarks_path = dataframe['Target landmarks'][current_id]
        status = dataframe['status'][current_id]

        # if status != "training":
        #     continue

        sizes = dataframe['Image size [pixels]'][current_id]
        sizes = sizes[:].split(", ")
        y_size = int(sizes[0][1:])
        x_size = int(sizes[1][:-1])

        params = dict()
        params['y_size'] = y_size
        params['x_size'] = x_size
        params['status'] = status
        params['output_path'] = results_path
        params['id'] = str(current_id)
        params['source_path'] = os.path.join(data_path, source_path)
        params['target_path'] = os.path.join(data_path, target_path)
        params['source_landmarks_path'] = os.path.join(data_path, source_landmarks_path)
        params['target_landmarks_path'] = os.path.join(data_path, target_landmarks_path)

        b_time = time.time()
        results = run_single(params)
        e_time = time.time()
        elapsed_time = (e_time - b_time) / 60
        print("Elapsed time: ", elapsed_time, " minutes.")

        transformed_source_landmarks_path = results['transformed_source_landmarks_path']
        if status == "training":
            i_tre = results['initial_tre']
            f_tre = results['resulting_tre']
            dataframe['Initial TRE Median'][current_id] = i_tre
            dataframe['Final TRE Median'][current_id] = f_tre

        dataframe['Execution time [minutes]'][current_id] = str(elapsed_time)
        dataframe['Warped source landmarks'][current_id] = transformed_source_landmarks_path
        dataframe.to_csv(output_csv_path)

def run_single(params):
    source_path = params['source_path']
    target_path = params['target_path']
    source_landmarks_path = params['source_landmarks_path']
    target_landmarks_path = params['target_landmarks_path']
    status = params['status']
    y_size = params['y_size']
    x_size = params['x_size']
    results_path = params['output_path']
    current_id = params['id']

    if not os.path.isdir(os.path.join(results_path, str(current_id))):
        os.mkdir(os.path.join(results_path, str(current_id)))

    source = utils.load_image(source_path)
    target = utils.load_image(target_path)

    source_landmarks = utils.load_landmarks(source_landmarks_path)
    if status == "training":
        print()
        print("Training case.")
        target_landmarks = utils.load_landmarks(target_landmarks_path)
    else:
        print()
        print("Evaluation case.")

    p_target, p_source, ia_target, ng_target, nr_target, i_u_x, i_u_y, u_x_nr, u_y_nr, warp_resampled_landmarks, warp_original_landmarks, return_dict = am.anhir_method(target, source)
    transformed_landmarks = warp_original_landmarks(source_landmarks)

    p_target = utils.normalize(p_target)
    p_source = utils.normalize(p_source)
    ia_target = utils.normalize(ia_target)
    ng_target = utils.normalize(ng_target)
    nr_target = utils.normalize(nr_target)

    p_target_i = utils.to_image(p_target)
    p_source_i = utils.to_image(p_source)
    ia_target_i = utils.to_image(ia_target)
    ng_target_i = utils.to_image(ng_target)
    nr_target_i = utils.to_image(nr_target)
    
    json_return_dict = json.dumps(return_dict)
    with open(os.path.join(results_path, str(current_id), "info.json"), "w") as f:
        f.write(json_return_dict)

    if status == "training":
        try:
            o_median = np.median(utils.rtre(source_landmarks, target_landmarks, x_size, y_size))
            r_median = np.median(utils.rtre(transformed_landmarks, target_landmarks, x_size, y_size))
            print("Initial rTRE: ", o_median)
            print("Resulting rTRE: ", r_median)
            string_to_save = "Initial TRE: " + str(o_median) + "\n" + "Resulting TRE: " + str(r_median)
            txt_path = os.path.join(results_path, str(current_id), "tre.txt")
            with open(txt_path, "w") as file:
                file.write(string_to_save)
        except:
            string_to_save = "Landmarks ERROR"
            txt_path = os.path.join(results_path, str(current_id), "tre_error.txt")
            with open(txt_path, "w") as file:
                file.write(string_to_save)     

    source_save_path = os.path.join(results_path, str(current_id), "source.png")
    target_save_path = os.path.join(results_path, str(current_id), "target.png")
    transformed_target_g_save_path = os.path.join(results_path, str(current_id), "target_ng.png")
    transformed_target_save_path = os.path.join(results_path, str(current_id), "target_nr.png")
    ia_target_save_path = os.path.join(results_path, str(current_id), "target_ia.png")
    sitk.WriteImage(p_source_i, source_save_path)
    sitk.WriteImage(p_target_i, target_save_path)
    sitk.WriteImage(ng_target_i, transformed_target_g_save_path)
    sitk.WriteImage(nr_target_i, transformed_target_save_path)
    sitk.WriteImage(ia_target_i, ia_target_save_path)

    transformed_source_landmarks_path = os.path.join(results_path, str(current_id), "transformed_source_landmarks.csv")
    utils.save_landmarks(transformed_source_landmarks_path, transformed_landmarks)

    return_dict = dict()
    return_dict['transformed_source_landmarks_path'] = os.path.join(str(current_id), "transformed_source_landmarks.csv")
    if status == "training":
        try:
            return_dict['initial_tre'] = o_median
            return_dict['resulting_tre'] = r_median
        except:
            return_dict['initial_tre'] = 0
            return_dict['resulting_tre'] = 0
    return return_dict



if __name__ == "__main__":
    run()