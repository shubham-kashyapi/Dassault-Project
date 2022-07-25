import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import json


def BS_plots(df_data, num_windows, lambda_param,save=False):

    '''
    BS_plots(df_data, num_windows, lambda_param,save=False)

    Parameters:
    df_data: Pandas dataframe where each column corresponds to a different time series
    num_windows range 1 to 200
    lambda_param range 0.00 to 1000.00
    to save the intervals as a json file give save=True
    
    Returns:
    List of tuples. Every tuple corresponds to one time series (column) in the csv file.
    Segmentation plots
    '''
    plots_to_return = []
    ts_arr = []
   
    for colname in list(df_data.columns):
        ts_arr = df_data[colname].dropna().to_numpy()
        ts_name = colname
        seg_model_obj = Bottom_Up_Segmentation(num_windows, lambda_param)
        intervals=seg_model_obj.train_model(ts_arr)
        print(intervals)
        fig=seg_model_obj.plot_segmentation(ts_name)
        plots_to_return.append([colname, fig])
        if save:
          with open(os.path.join("./", '{}.json'.format(ts_name)), 'w', encoding='utf-8') as f:
                json_str = json.dumps(intervals)
                f.write(json_str)  
        
    return plots_to_return



class Bottom_Up_Segmentation:
    def __init__(self, num_windows, lambda_param):
        self.__num_windows = num_windows
        self.__lambda_param = lambda_param
        return
    
    def __evaluate_norm_SSE(self, left_ind, right_ind):
        '''evaluates the SSE and returns the fit SSE and predicted data interval points'''
        time_steps = np.linspace(0, 1, right_ind-left_ind).reshape(-1,1)
        data_curr_interval = self.__data_ts[left_ind:right_ind]
        lin_mod = LinearRegression()
        lin_mod.fit(time_steps.reshape(-1,1), data_curr_interval)
        data_pred = lin_mod.predict(time_steps.reshape(-1,1))
        fit_sse = (right_ind-left_ind)/self.__num_pts*mean_squared_error(data_curr_interval, data_pred)
        return fit_sse, data_pred
    
    def train_model(self, data_ts):
        '''training of the model is done here'''
        self.__data_ts = data_ts
        self.__num_pts = data_ts.shape[0]
        break_pts = np.linspace(0, self.__num_pts, num = self.__num_windows+1, dtype = int)
        self.__intervals = [[break_pts[i], break_pts[i+1]] for i in range(self.__num_windows)]
        self.__sse_vals = []
        
        for interval in self.__intervals:
            sse_val, pred_vals = self.__evaluate_norm_SSE(interval[0], interval[1])
            self.__sse_vals.append(sse_val)
            
        sse_vals1 = self.__sse_vals        
        while True:
            tot_sse = np.sum(sse_vals1)
            #print(intervals, tot_sse)
            merged_intervals, merged_sse = [], []
            i = 0
            while i < len(self.__intervals)-1:
                interval_left, interval_right = self.__intervals[i], self.__intervals[i+1]
                len_left, len_right = interval_left[1]-interval_left[0], interval_right[1]-interval_right[0]
                cost_prev = sse_vals1[i] + sse_vals1[i+1] + self.__lambda_param*(1.0/len_left + 1.0/len_right)
                sse_curr, _ = self.__evaluate_norm_SSE(interval_left[0], interval_right[1])
                cost_curr = sse_curr + self.__lambda_param/(len_left+len_right)
                if cost_curr < cost_prev:
                    merged_intervals.append([interval_left[0], interval_right[1]])
                    merged_sse.append(sse_curr)
                    i += 2
                else:
                    merged_intervals.append(interval_left)
                    merged_sse.append(sse_vals1[i])
                    i += 1
            if i == len(self.__intervals)-1:
                merged_intervals.append(self.__intervals[i])
                merged_sse.append(sse_vals1[i])
            
            if len(self.__intervals) == len(merged_intervals):
                break
                
            self.__intervals = merged_intervals
            sse_vals1 = merged_sse
        
        return
        
            
    def plot_segmentation(self, ts_name): 
        '''ploting of the figure and the figure is returned'''       
        fig_plot=plt.figure()
        plt.scatter(np.arange(self.__num_pts), self.__data_ts, s = 0.5)
        plt.axvline(x = 0, c = 'r', linewidth = 1)
        for interval in self.__intervals:
            _, pred_vals = self.__evaluate_norm_SSE(interval[0], interval[1])
            plt.plot(np.arange(interval[0], interval[1]), pred_vals, c = 'k')
            plt.axvline(x = interval[1], c = 'r', linewidth = 1)
        plt.title('{}: Lambda parameter = {}'.format(ts_name, self.__lambda_param))
        plt.show()
        return fig_plot