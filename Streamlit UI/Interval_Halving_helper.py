import sys
import numpy as np
import pandas as pd
from Interval_Halving import Modified_Interval_Halving
from optimal_hyperparameter import AMethod
import time


def Interval_Halving_plots(df_data, var_search_space, window_search_space): 
    '''
    Parameters:
    df_data: Pandas dataframe where each column corresponds to a different time series
    var_search_space: tuple of min, max, steps for variance fraction
    window_search_space: tuple of min, max, steps for window size
    
    Returns:
    List of tuples. Every tuple corresponds to one time series (column) in the csv file.
    Segmentation plots
    Segmentation intervals (dict)
    '''
    plots_to_return = []
    
    for colname in list(df_data.columns):

        tseries = df_data[colname].dropna().to_numpy()
        err_series = np.var(tseries)
        # Hyperparameter tuning for variance fraction
        var_fracs = np.linspace(var_search_space[0], var_search_space[1], num = var_search_space[2])
        count_intervals = np.zeros(var_search_space[2], )
        for frac_ind, frac in enumerate(var_fracs):
            model = Modified_Interval_Halving(noise_err = frac*err_series, min_window_len = window_search_space[0], \
                                              title=colname, suppress_print=True)
            count_intervals[frac_ind] = model.train_model(tseries)

        elbow_model_var = AMethod()
        elbow_idx_var = elbow_model_var.get_elbow_point(var_fracs, count_intervals)
        opt_var_fraction = var_fracs[elbow_idx_var]

        # Hyperparameter tuning for minimum window length
        window_lens = np.linspace(window_search_space[0], window_search_space[1], num = window_search_space[2], dtype = int)
        count_intervals_window = np.zeros(window_search_space[2],)    
        for ind_window, window_len in enumerate(window_lens):
            model = Modified_Interval_Halving(noise_err = opt_var_fraction*err_series, min_window_len = window_len, \
                                              title = colname, suppress_print = True)
            count_intervals_window[ind_window] = model.train_model(tseries)  

        elbow_model_window = AMethod()
        elbow_idx_window = elbow_model_window.get_elbow_point(window_lens, count_intervals_window)
        opt_window = count_intervals_window[elbow_idx_window]

        # Running interval halving
        final_model = Modified_Interval_Halving(noise_err = opt_var_fraction*err_series, min_window_len = opt_window, \
                                                ylabel=colname, suppress_print=True)
        final_model.train_model(tseries)
        plot_name_interval = colname
        plot_obj_interval = final_model.plot_data_intervals()
        series_intervals = final_model.return_intervals()
        plots_to_return.append((plot_name_interval, plot_obj_interval, series_intervals))

    return plots_to_return
      
    

    