import streamlit as st
import numpy as np
import pandas as pd
from Interval_Halving_helper import Interval_Halving_plots
from AR_helper import AR_plots
from MVI_helper import MVI_plots
from Bottom_Up_Segmentation_helper import BS_plots
from Hierarchical_Interval_Halving_helper import HI_plots
#from TS_segmentation_helper import TS
import time

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_hi('Loading Anomaly Detection UI')
    st.title('Segmentation / Anomaly Detection from Time Series')
    
    option = st.sidebar.selectbox('Select the Model', ('Interval Halving', 'Auto Regression', 'Hurst Index','Bottom Up Segmentation','Hierarchical Segmentation','TS_Segmentation'))
    processing_option, benchmark_data, data_file, AR_Order = None, None, None, None
    
    # If chosen algorithm is AR, choose the AR order, processing mode and upload the benchmark model (if required)
    if option == 'Auto Regression':
        AR_Order = int(st.sidebar.number_input('Specify the order of the AR model (integer between 1 and 10)', 
                       min_value = 1, max_value = 10))
        outlier_removal = False
        if st.sidebar.checkbox('Remove outliers'):
            outlier_removal = True
        processing_option = st.sidebar.selectbox("Select whether or not to compare with a benchmark model.", ('No Benchmark', 'Benchmark'))
        if processing_option == 'Benchmark':
            benchmark_file = st.sidebar.file_uploader("\n\n Select a csv file containing the benchmark (coefficients as a single row or time                                                        series as a single column)")
            if benchmark_file is not None:
                benchmark_data = pd.read_csv(benchmark_file)
    
    if option == 'Bottom Up Segmentation':
        num_windows = int(st.sidebar.number_input('Specify the order of the num_windows (integer between 1 and 200 (select 100)   )', 
                       min_value = 1, max_value = 200))

        lambda_param = float(st.sidebar.number_input('Specify the order of the lambda_param (floating integer between 0 and 1000 (select 1)   )', 
                       min_value = 0.0, max_value = 1000.0))

    if option == 'Hierarchical Segmentation':
        alpha_val = float(st.sidebar.number_input('Specify the alpha value (integer between 0 and 10(select 0.05)   )', 
                       min_value = 0.00, max_value = 10.00))

    if option == 'TS_Segmentation':
        threshold = float(st.sidebar.number_input('Specify the alpha value (integer between 0 and 10(select 0.05)   )', 
                       min_value = 0, max_value = 100))
        window = float(st.sidebar.number_input('Specify the alpha value (integer between 0 and 10(select 0.05)   )', 
                       min_value = 5, max_value = 1000))


    if option == 'Interval Halving':
        # Specifying hyperparameter search space (In later versions, we can prompt the user to enter these values)
        var_min, var_max, var_steps = 1e-2, 1e-1, 10
        var_search_space = (var_min, var_max, var_steps)
        
        window_min, window_max, window_steps = 10, 100, 10
        window_search_space = (window_min, window_max, window_steps)
                

    data_file = st.file_uploader("Select a single file(csv) containing the Time Series Data. Each column should contain a header and a time                                   series.")
    if data_file is not None:
        st.header('Input')
        st.write(data_file.name)
        df_data = pd.read_csv(data_file)
        df_data.dropna()
        st.line_chart(df_data)
        if st.checkbox('Show Input Data'):
            st.write(df_data)

        if option == "Interval Halving": 
            # Calling the functions for hyperparameter tuning and generating optimal segmentation
            plots_to_show = Interval_Halving_plots(df_data, var_search_space, window_search_space)
            for (header_name, plot_display, segments_text) in plots_to_show:
                st.header(header_name)
                st.pyplot(plot_display)
                st.json(segments_text)
                st.write('________________________________________________________________________________')

        elif option == "Auto Regression":
            plots_to_show = AR_plots(df_data, benchmark_data, AR_Order, outlier_removal)
            for (header_name, plot_display) in plots_to_show:
                st.header(header_name)
                st.pyplot(plot_display)
                
        elif option == "Hurst Index":
            plots_to_show = MVI_plots(df_data)
            for (header_name, plot_display) in plots_to_show:
                st.header(header_name)
                st.pyplot(plot_display)
            
        elif option == "Bottom Up Segmentation":
            plots_to_show = BS_plots(df_data,num_windows, lambda_param)
            for (header_name, plot_display) in plots_to_show:
                st.header(header_name)
                st.pyplot(plot_display)

        elif option == "Hierarchical Segmentation":
            plots_to_show = HI_plots(df_data,alpha_val)
            for (header_name, plot_display) in plots_to_show:
                st.header(header_name)
                st.pyplot(plot_display)

        ''' elif option == "TS_Segmentation":
            plots_to_show = TS(df_data,window,threshold, display_intermediate = True)
            for (header_name, plot_display) in plots_to_show:
                st.header(header_name)
                st.pyplot(plot_display)'''
            
