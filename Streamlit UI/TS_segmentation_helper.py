import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TS_segmentation import *

def TS(df_data, windows, thershold,display_intermediate = True):
    ts_arr = []
    plots_to_return=[]
    for colname in list(df_data.columns):
        ts_arr = df_data[colname].dropna().to_numpy()
        seg_obj = Segmentation_Persistent_Homology('ordinal_partition')
        intervals_1 = seg_obj.get_segmentation(ts_arr, windows,thershold, display_intermediate = True)
        fig=intervals_1
        plots_to_return.append([colname, fig])
    
    return plots_to_return