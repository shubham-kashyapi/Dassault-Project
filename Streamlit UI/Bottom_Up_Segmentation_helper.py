import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bottom_Up_Segmentation import *

def BS_plots(df_data, num_windows, lambda_param):
    plots_to_return = []
    ts_arr = []
   
    for colname in list(df_data.columns):
        ts_arr = df_data[colname].dropna().to_numpy()

        ts_name = colname
        seg_model_obj = Bottom_Up_Segmentation(num_windows, lambda_param)
        seg_model_obj.train_model(ts_arr)
        fig=seg_model_obj.plot_segmentation(ts_name)
        


        plots_to_return.append([colname, fig])
        
        #plots_to_return = [(F'{colname} vs time', fig)]
    return plots_to_return