import sys
import os
import pathlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends import backend_pdf
from Interval_Halving import *
from optimal_hyperparameter import AMethod
from joblib import Parallel, delayed
from tqdm import tqdm
# Turn interactive plotting off
plt.ioff()

#######################################################################################
# Define functions to perform hyperparameter tuning and generate segmentation results
#######################################################################################
def hyperparam_tuning_var(time_series, series_name, var_data, curr_frac):
    model = Hierarchical_Interval_Halving(noise_err = curr_frac*var_data, min_window_len = window_lens_tuning[0], degrees = [0,1,2], \
                                      title = series_name, suppress_print = True, alpha_f_test = alpha_f_test_val)
    seg_count = model.train_model(time_series)
    return seg_count
    
def hyperparam_tuning_window(time_series, series_name, var_data, curr_window, opt_frac):
    model = Hierarchical_Interval_Halving(noise_err = opt_frac*var_data, min_window_len = int(curr_window), degrees = [0,1,2], \
                                      title = series_name, suppress_print = True, alpha_f_test = alpha_f_test_val)
    seg_count = model.train_model(time_series)
    return seg_count

def segmentation_analysis(time_series, series_name):
    err_data = np.var(time_series)
    ###########################################
    # Parameter tuning for variance fraction
    ###########################################
    
    count_segments_var = np.array([hyperparam_tuning_var(time_series, series_name, err_data, frac)
                                   for frac in var_fracs_tuning])
    
    opt_var_frac = var_fracs_tuning[AMethod().get_elbow_point(var_fracs_tuning, count_segments_var)]
    ###########################################
    # Parameter tuning for window length
    ###########################################          
    count_segments_window = np.array([hyperparam_tuning_window(time_series, series_name, 
                                     err_data, tuning_len, opt_var_frac) for tuning_len in window_lens_tuning])
    
    opt_window_len = window_lens_tuning[AMethod().get_elbow_point(window_lens_tuning, count_segments_window)]
    
    ###########################################
    # Fitting the optimal model
    ###########################################
    model = Hierarchical_Interval_Halving(noise_err = opt_var_frac*err_data, ylabel = series_name.split('_')[-1], \
                                      degrees = [0,1,2], min_window_len = opt_window_len, title = series_name, 
                                      suppress_print = True, alpha_f_test = alpha_f_test_val)
    model.train_model(time_series)        
    
    return model


def save_results(seg_model, series_data):
    # Saving the plot
    series_arr, series_name = series_data[1], series_data[0]
    plt.figure()
    plt.scatter(np.arange(series_arr.shape[0]), series_arr, c = 'tab:brown', s = 1)
    seg_model.plot_data_pred_intervals(linestyle = 'dotted', point_size = 1)
    plt.savefig(os.path.join(path_plots, '{}.png'.format(series_name)), dpi = 1000)
    
    # Saving the results as text
    res_intervals = seg_model.return_intervals()
    with open(os.path.join(path_text, '{}.json'.format(series_name)), 'w', encoding='utf-8') as f:
        json_str = json.dumps(res_intervals)
        f.write(json_str)     
    return

############################################################    
# Read data from file path passed as command line argument
# Create directories for storing results
############################################################
data_path = pathlib.Path(sys.argv[1])
result_path = pathlib.Path(sys.argv[2])
alpha_f_test_val = 0.05
print('Data path: {}'.format(data_path))
print('Result path: {}'.format(result_path))
dir_name = os.path.basename(os.path.normpath(data_path))
print(dir_name)
path_plots = os.path.join(result_path, dir_name, 'Plots/')
path_text = os.path.join(result_path, dir_name, 'Segmentation_text/')
print(path_plots, path_text)

os.makedirs(Path(path_plots), exist_ok = True)
os.makedirs(Path(path_text), exist_ok = True)
print("Output folders created")

'''
# Reading the data
print('Started reading the input data.')
sheet_to_df_map = pd.read_excel(Path(file_path), sheet_name = None)
print('Completed reading the input data.')
'''

################################################################################
# Specifying the hyperparameter search space (no intervention/change needed)
################################################################################
var_fracs_tuning = np.linspace(1e-4, 1e-2, num = 10)
window_lens_tuning = np.linspace(10, 100, num = 10, dtype = int)

#################################################################
# Reading all the time series
#################################################################
print('Started reading the input data.')
all_time_series = []
for csv_file in os.listdir(data_path)[1:]:
    file_path = os.path.join(data_path, csv_file)
    if pathlib.Path(file_path).suffix != '.csv':
        continue
    df = pd.read_csv(file_path)
    for col in df.columns:
        if col == 'TBLDR':
            continue
        # Read the data
        process_data = df[col].dropna().to_numpy()            
        len_data = process_data.shape[0]
        if len_data < 100: # Do not perform computation if time series is too short or empty
            continue
        ts_name = '{}_{}'.format(''.join(csv_file.split('.')[:-1]), col)
        all_time_series.append([ts_name, process_data])

print('Completed reading the input data.')

################################################
# Train the segmentation models parallelly
################################################
'''
print('Starting parallel execution for hyperparameter tuning and training the segmentation models.')
all_seg_models = Parallel(n_jobs = -1, verbose = 3)(delayed(segmentation_analysis)(ts, ts_name) for ts_name, ts in all_time_series)
print('Model training complete.')
'''
all_seg_models = []
for i in tqdm(range(len(all_time_series))):    
    all_seg_models.append(segmentation_analysis(all_time_series[i][1], all_time_series[i][0]))

#########################################
# Save the results (plot and json)
#########################################
print('Saving the results as images and json')
for curr_seg_model, curr_time_ser in zip(all_seg_models, all_time_series):
    save_results(curr_seg_model, curr_time_ser)
print('Saving complete.')       
    
        
       
            
            
            
            
            