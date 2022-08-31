import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import f
from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

class Hierarchical_Interval_Halving:
    def __init__(self, noise_err, xlabel = 'Time (sec)', ylabel = 'Pressure (mbar)', min_window_len = 100, \
                 alpha_f_test = 0.05, alpha_t_test = 0.05, degrees = [0,1,2], title = "", suppress_print = False):
        self.__noise_err = noise_err
        self.__xlabel = xlabel
        self.__ylabel = ylabel
        self.__min_window_len = min_window_len
        self.__title = title
        self.__suppress_print = suppress_print
        self.__alpha_f_test = alpha_f_test
        self.__alpha_t_test = alpha_t_test
        self.__degrees = degrees
        self.__degree_type_map = {1: 'Steady State (S)', 2: 'Degree 1 (D1)', 3: 'Degree 2 (D2)'}
        self.__interval_color_map = {'Steady State (S)': 'b', 'Degree 1 (D1)': 'g', 'Degree 2 (D2)': 'r', 'Noise (N)': 'k'}
        self.__data = np.array([])
        self.__predicted_vals = np.array([])
        self.__num_samples = 0
        self.__intervals = []
        self.__interval_types = [] # Whether the interval is steady state, has an identifiable trend or is noisy
        self.__temp_intervals = []
        self.__temp_interval_types = []
        self.__num_intervals = 0
        self.__reg_coeffs = []
        self.__reg_coeffs_covar = []
    
    def __find_intervals(self, left_ind, right_ind, degree):
        if self.__suppress_print == False:
            print('Interval: {}'.format((left_ind, right_ind)))
        curr_interval_len = (right_ind - left_ind + 1)
        data_curr_interval = self.__data[left_ind: right_ind+1,:]
        time_steps = np.linspace(0, 1, right_ind-left_ind+1).reshape(-1,1) # Time steps normalized to [0, 1] interval
        p_val = 0
        betas  = np.array([]) # Regression coefficients
        beta_var = np.array([])
        data_pred = np.zeros((curr_interval_len, ))
        
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
        # print(f_stat)
        p_val = 1 - f.cdf(f_stat, dof_numer, dof_denom)
        if self.__suppress_print == False:
            print('Degree = {}, p-val = {}'.format(degree, p_val))            
                
        if p_val >= self.__alpha_f_test:  # Good fit- Clear trend identified- No further splitting required
            self.__temp_intervals.append([left_ind, right_ind])
            interval_type = self.__degree_type_map[degree+1]
            self.__temp_interval_types.append(interval_type)
            #self.__reg_coeffs.append(betas)
            #self.__reg_coeffs_covar.append(beta_var)
            self.__predicted_vals[left_ind:right_ind+1] = data_pred.reshape(curr_interval_len, )
            
        elif curr_interval_len < 2*self.__min_window_len: # Poor fit and interval cannot be split further due to min window len
            self.__temp_intervals.append([left_ind, right_ind])
            self.__temp_interval_types.append('Noise (N)')
            #self.__reg_coeffs.append(betas)
            #self.__reg_coeffs_covar.append(beta_var)
            self.__predicted_vals[left_ind:right_ind+1] = self.__data[left_ind:right_ind+1].reshape(curr_interval_len, )
            
        else: # Recursive splitting
            mid = int((left_ind + right_ind)/2)
            self.__find_intervals(left_ind, mid, degree)
            self.__find_intervals(mid+1, right_ind, degree)
            
        return         
    
    def plot_data_intervals(self, marker = 'o', linestyle = 'solid', linewidth = 2):
        custom_lines = [Line2D([0], [0], marker = 'o', color='b'), Line2D([0], [0], marker = 'o', color='g'), 
                        Line2D([0], [0], marker = 'o', color='r'), Line2D([0], [0], marker = 'o', color='k')]
        
        for interval, interval_type in zip(self.__intervals, self.__interval_types):
            plt.plot(np.arange(interval[0], interval[1]+1), self.__data[interval[0] : interval[1]+1], \
                        c = self.__interval_color_map[interval_type], linewidth = linewidth, linestyle = linestyle)

        plt.legend(custom_lines, list(self.__interval_color_map.keys()))
        plt.title(self.__title)
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        #plt.show()  
        return
    
    def plot_data_pred_intervals(self, marker = 'o', linestyle = 'solid', point_size = 2):
        custom_lines = [Line2D([0], [0], marker = 'o', color='b'), Line2D([0], [0], marker = 'o', color='g'), 
                        Line2D([0], [0], marker = 'o', color='r'), Line2D([0], [0], marker = 'o', color='k')]
        
        for interval, interval_type in zip(self.__intervals, self.__interval_types):
            plt.scatter(np.arange(interval[0], interval[1]+1), self.__predicted_vals[interval[0] : interval[1]+1], \
                        c = self.__interval_color_map[interval_type], s = point_size)

        plt.legend(custom_lines, list(self.__interval_color_map.keys()))
        plt.title(self.__title)
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        # plt.show()  
        return
    
        
    def scaled_avg_global_err(self):        
        sage_sum, interval_tot = 0.0, 1e-10 # To avoid division by 0 error
        for interval, interval_type in zip(self.__intervals, self.__interval_types):
            if interval_type != 'Noise (N)':
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
        return self.__predicted_vals  
    
    def return_intervals(self):
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
            if interval_type != 'Noise (N)': # Steady-state or trend
                trend_removed_series[interval[0]: interval[1]+1] = self.__predicted_vals[interval[0]: interval[1]+1]
                
            else: # Noisy
                trend_removed_series[interval[0]: interval[1]+1] = self.__data[interval[0]: interval[1]+1, 0] - \
                                                                       np.mean(self.__data[interval[0]: interval[1]+1, 0])       
        
        return trend_removed_series
        
    def train_model(self, data): # data should be passed as 1d numpy array
        self.__data = data.reshape(-1, 1)
        self.__num_samples = self.__data.shape[0]
        self.__predicted_vals = np.array([0]*self.__num_samples, dtype = np.float64)
        self.__intervals, self.__interval_types = [[0, self.__num_samples-1]], ['Noise (N)']
        self.__temp_intervals, self.__temp_interval_types = [], []
        
        ########################################################
        # Trying degree fits in order (default: [0, 1, 2])
        ########################################################
        for degree in self.__degrees:            
            for seg, seg_type in zip(self.__intervals, self.__interval_types):
                if seg_type == 'Noise (N)':
                    self.__find_intervals(seg[0], seg[1], degree)
                else:
                    self.__temp_intervals.append(seg)
                    self.__temp_interval_types.append(seg_type)
                    
            self.__intervals, self.__interval_types = self.__temp_intervals, self.__temp_interval_types
            self.__temp_intervals, self.__temp_interval_types = [], []
        
        ##################################################################
        # Concatenation and finding the break points (Post processing)
        ##################################################################
        self.__break_points = []
        curr_seg = [self.__intervals[0], self.__interval_types[0]]
        for seg, segtype in zip(self.__intervals, self.__interval_types):
            if segtype == curr_seg[1]:
                curr_seg[0][1] = seg[1]
            else:
                self.__break_points.append(curr_seg)
                curr_seg = [seg, segtype]
        self.__break_points.append(curr_seg)                
        
        if self.__suppress_print == False:
            print('The intervals are: \n{}'.format(self.__intervals))
        self.__num_intervals = len(self.__break_points)
        return self.__num_intervals # Returns the number of intervals obtained by the Interval Halving Algorithm