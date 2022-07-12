import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Bottom_Up_Segmentation:
    def __init__(self, num_windows = 20, lambda_param = 1e3):
        self.__num_windows = num_windows
        self.__lambda_param = lambda_param
        return
    
    def __evaluate_norm_SSE(self, left_ind, right_ind):
        time_steps = np.linspace(0, 1, right_ind-left_ind).reshape(-1,1)
        data_curr_interval = self.__data_ts[left_ind:right_ind]
        lin_mod = LinearRegression()
        lin_mod.fit(time_steps.reshape(-1,1), data_curr_interval)
        data_pred = lin_mod.predict(time_steps.reshape(-1,1))
        fit_sse = (right_ind-left_ind)/self.__num_pts*mean_squared_error(data_curr_interval, data_pred)
        return fit_sse, data_pred
    
    def train_model(self, data_ts):
        self.__data_ts = data_ts
        self.__num_pts = data_ts.shape[0]
        break_pts = np.linspace(0, self.__num_pts, num = self.__num_windows+1, dtype = int)
        self.__intervals = [[int(break_pts[i]), int(break_pts[i+1])] for i in range(self.__num_windows)]
        self.__sse_vals = []
        
        for interval in self.__intervals:
            sse_val, pred_vals = self.__evaluate_norm_SSE(interval[0], interval[1])
            self.__sse_vals.append(sse_val)
            
        sse_vals1 = self.__sse_vals        
        while True:
            tot_sse = np.sum(sse_vals1)
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
            
        self.__sse_vals = sse_vals1
        return self.__intervals
        return
        
            
    def plot_segmentation(self, ts_name, xlabel = '', ylabel = '', path = None):        
        plt.figure()
        plt.scatter(np.arange(self.__num_pts), self.__data_ts, s = 0.5)
        plt.axvline(x = 0, c = 'r', linewidth = 1)
        for interval in self.__intervals:
            _, pred_vals = self.__evaluate_norm_SSE(interval[0], interval[1])
            plt.plot(np.arange(interval[0], interval[1]), pred_vals, c = 'k')
            plt.axvline(x = interval[1], c = 'r', linewidth = 1)
        plt.title('{}: Lambda parameter = {}\nSSE = {}'.format(ts_name, self.__lambda_param, np.sum(self.__sse_vals)))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if path is not None:
            plt.savefig(path, dpi = 1000)
        plt.show()
        return

            
        
        