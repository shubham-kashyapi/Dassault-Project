User Interface: Anomaly Detection from Time Series

Streamlit is a Python-based framework for UI/visualization apps.
Terminal command for installing streamlit: pip install streamlit
Terminal Command for launching the UI: streamlit run main.py
The UI will then open up in a browser window.

The following sample files have been provided in the folder "Sample files/"
(i) PSPCKL_1_4.csv 
(To provide your own input csv file, use the following format: Each column should contain a separate time series and should have a header
indicating the name of each series. The file may contain upto 10 time series.)
(ii) Benchmark_PSPCKL_coeffs.csv (Used for Auto Regression)
(iii) Benchmark_PSPCKL_Time_Series.csv (Used for Auto Regression)

From the left sidebar, choose the model: Interval Halving, Auto Regression, Minimum Variance Index.
_____________________________________________________________________________________________________________________________________________________________________________________________

(1) Interval Halving:
Input:
In the main panel, select a csv file containing the data. Each column should contain a separate time series and should have a header
indicating the name of each series. Sample file (i) may be used as an input.

Output: (Note that hyperparameter tuning is performed automatically, without requiring any intervention from the user. The optimal segmentation is computed and displayed to the user.)
The following outputs are displayed for every column in the csv file
(a) Plot indicating the steady state, transient and noisy regions for every time series
(b) Segmentation breakpoints displayed as a json string. 
Interpretation of the json: [[[0, 82], "Steady State (S)"], [[83, 123], "Noise (N)"], [[124, 204], "Transition (T)"], ..... ] means that the time series is in steady state from 
time index 0 to 82, noisy from 83 to 123, transition from 124 to 204, and so on.


Reference:
Dash, Maurya, Venkatasubramanian, Rengaswamy: A Novel Interval-Halving Framework For Automated Identification of Process Trends: AIChE Journal (January 2004 Vol. 50, No. 1)

_____________________________________________________________________________________________________________________________________________________________________________________________

(2) Auto Regression:
User Inputs / Settings:
(a) Order of the model: Increase or decrease its value in the left sidebar. Recommended value: 3
(b) Remove outliers checkbox: If checked, the analysis will be performed after removing outliers from the series. Otherwise, the outliers will not be discarded, considering them to be an important part of the data for further analysis.
(c) Benchmark/ No benchmark dropdown: Choose whether to compare the computed models with the benchmark model/ model obtained from the benchmark series.
(d) Benchmark model or Benchmark Series: If chosen to compare with the benchmark at (c), the UI will ask the user to enter either of the following:
- A CSV file containing autoregressive model coefficients. It will be used as the benchmark for comparing models obtained from other series. The number of coefficients in the CSV file should be consistent with the order of the model chosen at (a). Here sample file (ii) can be used as an input if the AR order is chosen as 3.
- A CSV file containing a time series to be used as a benchmark. An autoregressive model calculated from this time series will be used as a benchmark to compare models obtained from the other series. Here sample file (iii) can be used as an input.
(e) Variable data: In the main panel, UI will ask the user to insert the CSV file containing the time series of the variable to be used for analysis. Here sample file (i) may be used as an input.

All the input CSV files are supposed to have a header.

Output:
(a) A bar plot showing angle made by models obtained from each time series with the average/benchmark model.
(b) A heatmap showing angles between the models calculated from each pair of time series (only if "No benchmark" option is chosen in (c)).

____________________________________________________________________________________________________________________________________________________________________________________________

(3) Minimum Variance Index:
Input:
In the main panel, select a csv file containing the data. Each column should contain a separate time series and should have a header
indicating the name of each series. Sample file (i) may be used as an input.

Output:
Single scatter plot indicating the Minimum Variance Index/Hurst Index for all time series. The time series are named from 1 to n if there are n columns in the input file.

Reference:
B. Srinivasan, T. Spinner, R. Rengaswamy: Control loop performance assessment using detrended fluctuation analysis (DFA): Automatica 48 (2012) 1359-1363

_____________________________________________________________________________________________________________________________________________________________________________________________


