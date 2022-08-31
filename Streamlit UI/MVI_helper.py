import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Minimum_Variance_Index import Scaling_Exponents

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