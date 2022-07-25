import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat
from typing import Tuple
import matplotlib.pyplot as plt



def MVI_plots(df_data):
    MVI_vals = []
    for colname in list(df_data.columns):
        tseries = df_data[colname].dropna().to_numpy()
        _, mvi = Scaling_Exponents(tseries)
        MVI_vals.append(mvi)
        
    fig = plt.figure()
    plt.xticks(np.arange(1, len(df_data.columns)+1))
    plt.scatter(np.arange(1, len(df_data.columns)+1), np.array(MVI_vals))
    plt.xlabel('Case ID')
    plt.ylabel('MVI (eta)')
    plt.grid()
    plots_to_return = [('Minimum Variance Index vs Case ID', fig)]
    return plots_to_return





def Scaling_Exponents(y_series: 'np.ndarray') -> Tuple['np.float64', 'np.float64']:
    '''
    Input- y_series: 1D numpy array of floats (time series) - Should not contain NaN values, Length should be greater than 100
    Returns- alpha (Generalized Hurst exponent), eta (Generalized scaling exponent)- np.float64
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
    window_lens = np.arange(10, int(np.floor(num_samples/4)), 10) 
    num_window_lens = window_lens.shape[0]
    rms_fluctuations = np.zeros((num_window_lens, ))
    
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
            Xdata, ydata = indices[left_ind: right_ind].reshape(-1,1), y_integrated[left_ind: right_ind]
            lin_mod = LinearRegression().fit(Xdata, ydata)
            y_pred[left_ind: right_ind] = lin_mod.predict(Xdata)
            
        rms_fluctuations[ind_len_window] = np.sqrt(mean_squared_error(y_integrated, y_pred))
    
    # Linear regression of rms fluctuations vs window lengths
    lin_reg = LinearRegression().fit(np.log(window_lens).reshape(-1,1), np.log(rms_fluctuations))
    alpha = lin_reg.coef_[0] # Generalized Hurst exponent
    eta = (2*alpha) if alpha <= 0.5 else (1.5 - alpha) # Generalized scaling exponent
    
    return alpha, eta
        