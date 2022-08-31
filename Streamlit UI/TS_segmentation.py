import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gtda.graphs import TransitionGraph
from gtda.time_series import SingleTakensEmbedding, TakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape
from gtda.diagrams import PersistenceEntropy
from scipy.signal import find_peaks
from scipy.sparse.csgraph import shortest_path

class Segmentation_Persistent_Homology:
    
    def __init__(self, homology_type):
        '''
        Parameters:
        homology_type: 'ordinal_partition' or 'takens_embedding'
        point_summary_type: 'max_persistence', 'periodicity_score', 
                            'homology_classes_by_graph_order', 'norm_persistent_entropy'
        
        Returns:
        None
        '''
        self.homology_type = homology_type
        return
    
    def kernel_dist(self, persis1, persis2, sig = 1.0):
        kernel_func_12, kernel_func_11, kernel_func_22 = 0.0, 0.0, 0.0
        for pt1 in range(persis1.shape[1]):
            for pt2 in range(persis2.shape[1]):
                dist_1 = (persis1[0, pt1, 0]-persis2[0, pt2, 0])**2 + (persis1[0, pt1, 1]-persis2[0, pt2, 1])**2
                dist_2 = (persis1[0, pt1, 0]-persis2[0, pt2, 1])**2 + (persis1[0, pt1, 1]-persis2[0, pt2, 0])**2
                kernel_func_12 = kernel_func_12 + np.exp(-1.0/(8*sig)*dist_1) + np.exp(-1.0/(8*sig)*dist_2)        
        kernel_func_12 = 1.0/(8.0*np.pi*sig)*kernel_func_12
        
        for pt1 in range(persis1.shape[1]):
            for pt2 in range(persis1.shape[1]):
                dist_1 = (persis1[0, pt1, 0]-persis1[0, pt2, 0])**2 + (persis1[0, pt1, 1]-persis1[0, pt2, 1])**2
                dist_2 = (persis1[0, pt1, 0]-persis1[0, pt2, 1])**2 + (persis1[0, pt1, 1]-persis1[0, pt2, 0])**2
                kernel_func_11 = kernel_func_11 + np.exp(-1.0/(8*sig)*dist_1) + np.exp(-1.0/(8*sig)*dist_2)
        kernel_func_11 = 1.0/(8.0*np.pi*sig)*kernel_func_11
                
        for pt1 in range(persis2.shape[1]):
            for pt2 in range(persis2.shape[1]):
                dist_1 = (persis2[0, pt1, 0]-persis2[0, pt2, 0])**2 + (persis2[0, pt1, 1]-persis2[0, pt2, 1])**2
                dist_2 = (persis2[0, pt1, 0]-persis2[0, pt2, 1])**2 + (persis2[0, pt1, 1]-persis2[0, pt2, 0])**2
                kernel_func_22 = kernel_func_22 + np.exp(-1.0/(8*sig)*dist_1) + np.exp(-1.0/(8*sig)*dist_2)
        kernel_func_22 = 1.0/(8.0*np.pi*sig)*kernel_func_22
        
        dist_ker = (kernel_func_11 + kernel_func_22 -2*kernel_func_12)**0.5
        return dist_ker
        
        
    
    def get_persistence_diagram(self, ts_window):
        '''
        Parameters:
        ts_window: 1d numpy array (for ordinal_partition) ; 1d or 2d numpy array (for takens_embedding)
        
        Returns:
        None
        '''
        homology_dimensions = [1]
        
        if self.homology_type == 'ordinal_partition':           
            tau_curr, d_curr = 2, 5 # Hard-coded values (TO FIX later)
            indices = np.array([np.arange(i, i+d_curr*tau_curr, tau_curr) 
                                for i in range(ts_window.shape[0]-(d_curr-1)*tau_curr)])
            X_perm_window = ts_window[indices]
            X_tg = TransitionGraph().fit_transform([X_perm_window])[0]
            dist_matrix = shortest_path(X_tg)
            persis = VietorisRipsPersistence(metric = 'precomputed', homology_dimensions = homology_dimensions, 
                                                n_jobs = -1).fit_transform([dist_matrix])
            persis[np.isinf(persis)] = 1e2
            
        elif self.homology_type == 'takens_embedding':
            max_embedding_dimension, stride = 5, 1 # Hard-coded values (TO FIX later)
            max_time_delay = ts_window.shape[0]//(max_embedding_dimension + 1)

            TS_embedded = SingleTakensEmbedding(parameters_type= 'search', time_delay = max_time_delay, 
                                                dimension = max_embedding_dimension,
                                                stride = stride).fit_transform(ts_window)
            
            crshp_embed = TS_embedded[None, :, :]
            
            persis = VietorisRipsPersistence(homology_dimensions = homology_dimensions,
                                                n_jobs = -1).fit_transform(crshp_embed)
            persis[np.isinf(persis)] = 1e2
        
        else:
            raise ValueError("Invalid value of self.homology_type")
            
        return persis
            
    
    def get_segmentation(self, ts_data, window_len, threshold = 1.0, display_intermediate = True):
        '''
        Parameters:
        ts_data: 1d numpy array (for ordinal_partition) ; 1d or n-dim numpy array (for takens_embedding)
        window_len: integer
        display_intermediate (bool): Whether to display the segmentation at intermediate iterations
        
        Returns:
        None
        '''
        intervals = [[start_ind, start_ind + window_len] for start_ind in range(0, ts_data.shape[0], window_len)]
        passes_count = 0
        
        while True:
            if display_intermediate:
                print('Pass number = {}'.format(passes_count))
                print(intervals)
                plt.figure()
                plt.scatter(np.arange(ts_data.shape[0]), ts_data, s = 1)
                plt.axvline(x = 0, c = 'r')
                for interval in intervals:
                    plt.axvline(x = interval[1], c = 'r')
                plt.show()
            
            persisDiagrams = []
            for interval in intervals:
                persisDiagrams.append(self.get_persistence_diagram(ts_data[interval[0]: interval[1]]))
                
            intervals_merged = []
            
            ind = 0
            while ind < len(intervals)-1:
                dist_adj = self.kernel_dist(persisDiagrams[ind], persisDiagrams[ind+1])
                if(dist_adj < threshold):
                    intervals_merged.append([intervals[ind][0], intervals[ind+1][1]])
                    ind += 2
                else:
                    intervals_merged.append([intervals[ind][0], intervals[ind][1]])
                    ind += 1
                    
            if ind == len(intervals)-1:
                intervals_merged.append([intervals[ind][0], intervals[ind][1]])
                
            if len(intervals) == len(intervals_merged):
                break
            
            intervals = intervals_merged
            
            passes_count += 1
                
        plt.figure()
        plt.scatter(np.arange(ts_data.shape[0]), ts_data, s = 1)
        plt.axvline(x = 0, c = 'r')
        for interval in intervals:
            plt.axvline(x = interval[1], c = 'r')
        plt.show()
        
        return intervals