from AR_Model import AR_Approach
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

def AR_plots(Input_Data, Bench_Data, Ar_Order, fill_outlier):
    # Default parameters for AR model
    Mm_Fra = 0.1
    
    # Computing the AR coefficients
    Calc_Models = []        
    for case_n in range(Input_Data.shape[1]):
        AR = AR_Approach(Ar_Order, Mm_Fra, fill_outlier)
        AR.Get_Model(Input_Data.iloc[:,case_n].dropna())
        Calc_Models.append(np.array(AR.Mod))

    Calc_Models = np.array(Calc_Models)
    
    plots_to_return = []
    if Bench_Data is not None: # Bar plot of angles with benchmark model
        Bench_Mod = None
        if len(Bench_Data) == 1: # Coefficients are directly specified
            Bench_Mod = np.array(Bench_Data.iloc[0, 0:], dtype = float)
            if Bench_Mod.shape[0] != (Ar_Order+1):
                raise ValueError("Mismatch between the specified AR Order and the number of coefficients in the benchmark model.")
        else: #Coefficients need to be computed from time series
            AR = AR_Approach(Ar_Order, Mm_Fra, fill_outlier)
            AR.Get_Model(Bench_Data.iloc[:,0].dropna())
            Bench_Mod = np.array(AR.Mod)
        Bench_Angles = np.zeros((Input_Data.shape[1],), dtype = float)
        for i in range(Input_Data.shape[1]):
            Bench_Angles[i] = np.rad2deg(np.arccos(np.clip((Calc_Models[i,:] @ Bench_Mod)/
                                         (np.linalg.norm(Calc_Models[i,:])*np.linalg.norm(Bench_Mod)), -1, 1)))
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(list(Input_Data.columns), Bench_Angles)
        plt.xlabel('Case ID')
        plt.ylabel('Angle (degrees)')
        plots_to_return.append(('Angles with Benchmark Model', fig))
        
    else: 
        # Bar plot of angles with average model
        Avg_Mod = np.mean(Calc_Models, axis = 0)
        Avg_Angles = np.zeros((Input_Data.shape[1],), dtype = float)
        for i in range(Input_Data.shape[1]):
            Avg_Angles[i] = np.rad2deg(np.arccos(np.clip((Calc_Models[i,:] @ Avg_Mod)/
                                         (np.linalg.norm(Calc_Models[i,:])*np.linalg.norm(Avg_Mod)), -1, 1)))
            
        fig = plt.figure()
        fig.add_axes([0,0,1,1])
        plt.bar(list(Input_Data.columns), Avg_Angles)
        plt.xlabel('Case ID')
        plt.ylabel('Angle (degrees)')
        plots_to_return.append(('Angles with Average Model', fig))
        
        # Pairwise heatmap of computed models
        Model_Angles = np.zeros((Input_Data.shape[1], Input_Data.shape[1]), dtype = float)
        for i in range(Input_Data.shape[1]):
            for j in range(i+1, Input_Data.shape[1]):
                angle_calc = np.rad2deg(np.arccos(np.clip((Calc_Models[i,:] @ Calc_Models[j,:])/
                                                  (np.linalg.norm(Calc_Models[i,:])*np.linalg.norm(Calc_Models[j,:])), -1, 1)))
                Model_Angles[i, j] = angle_calc
                Model_Angles[j, i] = angle_calc
        
        sns.set(font_scale=1.3)
        plt.figure()
        ax = sns.heatmap(np.around(Model_Angles, decimals = 2), cmap = "OrRd", robust=True, annot=True, 
                         xticklabels = list(Input_Data.columns), yticklabels = list(Input_Data.columns))
        plots_to_return.append(('Pairwise Heatmap', ax.get_figure()))
        
    return plots_to_return
        
    