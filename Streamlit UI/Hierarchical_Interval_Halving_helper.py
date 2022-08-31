import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Hierarchical_Interval_Halving  import *
from optimal_hyperparameter import AMethod

def HI_plots(df_data,alpha_val):
    ts_arr = []
    plots_to_return = []
    for colname in list(df_data.columns):
        ts_arr = df_data[colname].dropna().to_numpy()
        ts_name = colname

        #to be hard coded later into UI
        var_fracs_tuning = np.linspace(1e-2, 2e-1, num = 10)
        window_lens_tuning = np.linspace(10, 200, num = 10, dtype = int)
        alpha_f_test_val = alpha_val


        def hyperparam_tuning_var(time_series, series_name, var_data, curr_frac):
            model = Hierarchical_Interval_Halving(noise_err = curr_frac*var_data, min_window_len = window_lens_tuning[0], degrees = [0,1], \
                                            title = series_name, suppress_print = True, alpha_f_test = alpha_f_test_val)
            seg_count = model.train_model(time_series)
            return seg_count
            
        def hyperparam_tuning_window(time_series, series_name, var_data, curr_window, opt_frac):
            model = Hierarchical_Interval_Halving(noise_err = opt_frac*var_data, min_window_len = int(curr_window), degrees = [0,1], \
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
                                            degrees = [0,1], min_window_len = opt_window_len, title = series_name, 
                                            suppress_print = True, alpha_f_test = alpha_f_test_val)
            model.train_model(time_series)        
            
            return model


        
        seg_model = segmentation_analysis(ts_arr, ts_name)
        fig=plt.figure()
        seg_model.plot_data_pred_intervals(linestyle = 'dotted', point_size = 2)
        plt.show()
        plots_to_return.append([colname, fig])
    
    
    return plots_to_return