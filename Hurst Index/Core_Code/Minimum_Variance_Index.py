import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Tuple
import matplotlib.pyplot as plt

def Scaling_Exponents(y_series: 'np.ndarray', degree_fit = 2, min_window = 10, step_size = 10, display_plots = True) -> Tuple['np.float64', 'np.float64']:
    '''
    Input- y_series: 1D numpy array of floats (time series) - Should not contain NaN values, Length should be greater than 100
    Returns- alpha (Hurst index), eta (Generalized scaling exponent)- np.float64
    '''
    # Checking if the input is valid
    if type(y_series) != np.ndarray:
        raise TypeError("Given input is not a numpy array")
    elif y_series.ndim != 1:
        raise ValueError("Given numpy array is not 1D")
    num_samples = y_series.shape[0]
    if num_samples < 100:
        raise ValueError("Length of the time series is less than 100")
    
    # Window lengths ranging from 10 to num_samples/4 in steps of 10
    window_lens = np.arange(min_window, int(np.floor(num_samples/4)), step_size) 
    num_window_lens = window_lens.shape[0]
    rms_fluctuations = np.zeros((num_window_lens, ))
    
    if display_plots:
        fig, axs = plt.subplots(1, 2, figsize = (16, 6))
        plt.subplot(1, 2, 1)
    
    for ind_len_window, len_window in enumerate(window_lens):
        num_windows = int(np.floor(num_samples/len_window))    # Number of whole windows in the time series
        samples_used = int(len_window*num_windows)             # Number of samples to be used
        #print(samples_used)
        y_centered = y_series[:samples_used] - np.mean(y_series[:samples_used]) # Mean centering        
        y_integrated = np.cumsum(y_centered)                   # Computing the integrated time series
        indices = np.arange(samples_used)
        # Applying linear regression to each window and computing the RMS error
        y_pred = np.zeros((samples_used, ))
        for ind_window in range(num_windows):
            left_ind, right_ind = ind_window*len_window, (ind_window+1)*len_window # Left and right ends of the window
            Xdata, ydata = indices[left_ind: right_ind], y_integrated[left_ind: right_ind]
            Xdata_trans = PolynomialFeatures(degree_fit).fit_transform(Xdata.reshape(-1,1))
            lin_mod = LinearRegression(fit_intercept=False).fit(Xdata_trans, ydata)
            y_pred[left_ind: right_ind] = lin_mod.predict(Xdata_trans)
            
            
        rms_fluctuations[ind_len_window] = np.sqrt(mean_squared_error(y_integrated, y_pred))
        if display_plots:
            plt.scatter(np.arange(1, y_integrated.shape[0]+1), y_integrated.flatten(), color = 'b', s = 1)
            plt.scatter(np.arange(1, y_pred.shape[0]+1), y_pred.flatten(), color = 'r', s = 1)
            #plt.pause(0.05)
            fig.canvas.draw()
        
        
    
    # Linear regression of rms fluctuations vs window lengths
    lin_reg = LinearRegression(fit_intercept = False).fit(np.log(window_lens).reshape(-1,1), np.log(rms_fluctuations))
    alpha = lin_reg.coef_[0] # Hurst index (using DFA)
    eta = (2*alpha) if alpha <= 0.5 else (1.5 - alpha) # Generalized scaling exponent
    
    if display_plots:
            plt.subplot(1, 2, 2)
            plt.scatter(np.log(window_lens), np.log(rms_fluctuations), color = 'b')
            fig.canvas.draw()
            #plt.show()            
    
    return alpha, eta
        