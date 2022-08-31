import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import f
from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
from . import optimal_hyperparameter
import time
import os
import json


def Interval_Halving_plots(df_data,save=False): 

    '''
    Parameters:
    df_data: Pandas dataframe where each column corresponds to a different time series
    var_search_space: tuple of min, max, steps for variance fraction
    window_search_space: tuple of min, max, steps for window size

    Interval_Halving_plots(df_data,save=False) give save=true to save the output intervals into a json file.
    
    Returns:
    List of tuples. Every tuple corresponds to one time series (column) in the csv file.
    Segmentation plots
    Segmentation intervals (dict)
    '''
    var_min, var_max, var_steps = 1e-2, 1e-1, 10
    var_search_space = (var_min, var_max, var_steps)
        
    window_min, window_max, window_steps = 10, 100, 10
    window_search_space = (window_min, window_max, window_steps)
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

        elbow_model_var = optimal_hyperparameter.AMethod()
        elbow_idx_var = elbow_model_var.get_elbow_point(var_fracs, count_intervals)
        opt_var_fraction = var_fracs[elbow_idx_var]

        # Hyperparameter tuning for minimum window length
        window_lens = np.linspace(window_search_space[0], window_search_space[1], num = window_search_space[2], dtype = int)
        count_intervals_window = np.zeros(window_search_space[2],)    
        for ind_window, window_len in enumerate(window_lens):
            model = Modified_Interval_Halving(noise_err = opt_var_fraction*err_series, min_window_len = window_len, \
                                              title = colname, suppress_print = True)
            count_intervals_window[ind_window] = model.train_model(tseries)  

        elbow_model_window = optimal_hyperparameter.AMethod()
        elbow_idx_window = elbow_model_window.get_elbow_point(window_lens, count_intervals_window)
        opt_window = count_intervals_window[elbow_idx_window]

        # Running interval halving
        final_model = Modified_Interval_Halving(noise_err = opt_var_fraction*err_series, min_window_len = opt_window, \
                                                ylabel=colname, suppress_print=True)
        final_model.train_model(tseries)
        plot_name_interval = colname
        plot_obj_interval = final_model.plot_data_intervals()
        series_intervals = final_model.return_intervals()
        print(series_intervals)
        plots_to_return.append((plot_name_interval, plot_obj_interval))
        if save:
            with open(os.path.join("./", '{}.json'.format(plot_name_interval)), 'w', encoding='utf-8') as f:
                json_str = json.dumps(series_intervals)
                f.write(json_str)

    return plots_to_return


class Modified_Interval_Halving:
    
    def __init__(self, noise_err, xlabel = 'Time (sec)', ylabel = 'Pressure (mbar)', min_window_len = 100, \
                 alpha_f_test = 0.05, alpha_t_test = 0.05, title = "", suppress_print = False):
        self.__noise_err = noise_err
        self.__xlabel = xlabel
        self.__ylabel = ylabel
        self.__min_window_len = min_window_len
        self.__title = title
        self.__suppress_print = suppress_print
        self.__alpha_f_test = alpha_f_test
        self.__alpha_t_test = alpha_t_test
        self.__degrees = [0, 1, 2]
        self.__data = np.array([])
        self.__predicted_vals = np.array([])
        self.__num_samples = 0
        self.__intervals = []
        self.__interval_types = [] # Whether the interval is staedy state, has an identifiable trend or is noisy
        self.__num_intervals = 0
        self.__reg_coeffs = []
        self.__reg_coeffs_covar = []
    
    def __find_intervals(self, left_ind, right_ind):
        ''' We find the time interval here.Time steps normalized to [0, 1] interval.betas are Regression coefficients
        '''

        if self.__suppress_print == False:
            print('Interval: {}'.format((left_ind, right_ind)))
        curr_interval_len = (right_ind - left_ind + 1)
        data_curr_interval = self.__data[left_ind: right_ind+1,:]
        time_steps = np.linspace(0, 1, right_ind-left_ind+1).reshape(-1,1) # Time steps normalized to [0, 1] interval
        p_val = 0
        betas  = np.array([]) # Regression coefficients
        beta_var = np.array([])
        data_pred = np.zeros((curr_interval_len, ))
        
        for degree in self.__degrees:
            time_steps_trans = PolynomialFeatures(degree).fit_transform(time_steps) # Polynomial features
            lin_mod = LinearRegression()
            lin_mod.fit(time_steps_trans, data_curr_interval)
            data_pred = lin_mod.predict(time_steps_trans)
            fit_mse = mean_squared_error(data_curr_interval, data_pred)
            betas = lin_mod.coef_[0]
            beta_var = self.__noise_err*np.linalg.inv(np.dot(time_steps_trans.T, time_steps_trans))

            # F-test to compare with noise variance
            dof_numer, dof_denom = (curr_interval_len - 1), (self.__num_samples - 1)
            f_stat = fit_mse/self.__noise_err
            # print('Numer = {}, Denom = {}'.format(fit_mse, self.__noise_err))
            #print(f_stat)
            p_val = 1 - f.cdf(f_stat, dof_numer, dof_denom)
            if self.__suppress_print == False:
                print('Degree = {}, p-val = {}'.format(degree, p_val))            
        
            if p_val >= self.__alpha_f_test: # Do not need to fit a higher degree polynomial
                break
                
        if p_val >= self.__alpha_f_test:  # Good fit- Clear trend identified- No further splitting required
            self.__intervals.append([left_ind, right_ind])
            interval_type = 'Transition (T)' if betas.shape[0] > 1 else 'Steady State (S)'
            self.__interval_types.append(interval_type)
            self.__reg_coeffs.append(betas)
            self.__reg_coeffs_covar.append(beta_var)
            self.__predicted_vals[left_ind:right_ind+1] = data_pred.reshape(curr_interval_len, )
            
        elif curr_interval_len < 2*self.__min_window_len: # Poor fit-Noisy region-Cannot be split further due to min window len
            self.__intervals.append([left_ind, right_ind])
            self.__interval_types.append('Noise (N)')
            self.__reg_coeffs.append(betas)
            self.__reg_coeffs_covar.append(beta_var)
            self.__predicted_vals[left_ind:right_ind+1] = self.__data[left_ind:right_ind+1].reshape(curr_interval_len, )
            
        else: # Recursive splitting
            mid = int((left_ind + right_ind)/2)
            self.__find_intervals(left_ind, mid)
            self.__find_intervals(mid+1, right_ind)
            
        return         
    
    def plot_data_intervals(self):
        ''' ploting is done with respective coloring Steady State (S): 'blue', Transition (T): 'red', Noise (N): 'black'
        '''
        custom_lines = [Line2D([0], [0], marker = 'o', color='b'), Line2D([0], [0], marker = 'o', color='r'),
                        Line2D([0], [0], marker = 'o', color='k')]

        interval_color_map = {'Steady State (S)': 'b', 'Transition (T)': 'r', 'Noise (N)': 'k'}
        
        fig_plot = plt.figure()
        for interval, interval_type in zip(self.__intervals, self.__interval_types):
            plt.scatter(np.arange(interval[0], interval[1]+1), self.__data[interval[0] : interval[1]+1], \
                        color = interval_color_map[interval_type], s = 0.5)

        plt.legend(custom_lines, ['Steady State (S)', 'Transition (T)', 'Noise (N)'])
        plt.title(self.__title)
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        # plt.show()  
        return fig_plot
    
    def plot_data_pred_intervals(self):
        ''' ploting is done with predictive intervals and returns the plot with respective coloring Steady State (S): 'blue', Transition (T): 'red', Noise (N): 'black'
        '''
        custom_lines = [Line2D([0], [0], marker = 'o', color='b'), Line2D([0], [0], marker = 'o', color='r'),
                        Line2D([0], [0], marker = 'o', color='k')]

        interval_color_map = {'Steady State (S)': 'b', 'Transition (T)': 'r', 'Noise (N)': 'k'}
        
        fig_plot = plt.figure()
        for interval, interval_type in zip(self.__intervals, self.__interval_types):
            plt.scatter(np.arange(interval[0], interval[1]+1), self.__predicted_vals[interval[0] : interval[1]+1], \
                        color = interval_color_map[interval_type], s = 0.5)

        plt.legend(custom_lines, ['Steady State (S)', 'Transition (T)', 'Noise (N)'])
        plt.title(self.__title)
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        # plt.show()  
        return fig_plot
    
        
    def scaled_avg_global_err(self):        
        sage_sum, interval_tot = 0.0, 1e-10 # To avoid division by 0 error
        for interval, interval_type in zip(self.__intervals, self.__interval_types):
            if interval_type != 'Noise':
                interval_len = (interval[1]-interval[0]+1)
                sage_sum += interval_len*mean_squared_error(self.__data[interval[0] : interval[1]+1], \
                                           self.__predicted_vals[interval[0]: interval[1]+1])
                interval_tot += interval_len
        
        sage = sage_sum/interval_tot
        return sage
    
    def frac_noise(self):
        '''
        Gives the fraction of time series points which correspond to the noisy region
        '''
        print(self.__intervals)
        noise_len = 0.0
        for inter_indices, inter_type in zip(self.__intervals, self.__interval_types):
            if inter_type == 'Noise (N)':
                noise_len += (inter_indices[1] - inter_indices[0])
                
        return (noise_len/self.__num_samples)
    
    def pred_data(self):
        '''Returns predicted data interval points'''
        return self.__predicted_vals  
    
    def return_intervals(self):
        '''Returns predicted merged data interval points'''
        self.__merged_intervals = []
        start_ind, end_ind = 0, 0
        curr_type = self.__interval_types[0]
        for interval, interval_type in zip(self.__intervals, self.__interval_types):
            if interval_type == curr_type:
                end_ind = interval[1]
            else:
                self.__merged_intervals.append(([start_ind, end_ind], curr_type))
                start_ind, end_ind = interval[0], interval[1]
                curr_type = interval_type
               
        self.__merged_intervals.append(([start_ind, end_ind], curr_type))            
        return self.__merged_intervals
    
    def remove_trends(self):
        '''
        (Used for Hurst Index computation)
        For the steady-state segments, trends (degree 1 and 2), the predicted value is
        subtracted from the data. For the noisy segments, the mean of the segment is 
        subtracted from the data.
        
        The returned time series contains only noise (centered around 0) and no trends
        '''
        trend_removed_series = np.zeros((self.__num_samples), dtype = float)
        
        for interval, interval_type in zip(self.__intervals, self.__interval_types):
            if interval_type != 'Noise': # Steady-state or trend
                trend_removed_series[interval[0]: interval[1]+1] = self.__predicted_vals[interval[0]: interval[1]+1]
                
            else: # Noisy
                trend_removed_series[interval[0]: interval[1]+1] = self.__data[interval[0]: interval[1]+1, 0] - \
                                                                       np.mean(self.__data[interval[0]: interval[1]+1, 0])       
        
        return trend_removed_series
        
    def train_model(self, data): # data should be passed as 1d numpy array
        '''training of the model is doen here.Returns the number of intervals obtained by the Interval Halving Algorithm'''
        self.__data = data.reshape(-1, 1)
        self.__num_samples = self.__data.shape[0]
        self.__predicted_vals = np.array([0]*self.__num_samples, dtype = np.float64)
        self.__find_intervals(0, self.__num_samples-1)               
        
        if self.__suppress_print == False:
            print('The intervals are: \n{}'.format(self.__intervals))
        self.__num_intervals = len(self.__intervals)
        return self.__num_intervals # Returns the number of intervals obtained by the Interval Halving Algorithm
               