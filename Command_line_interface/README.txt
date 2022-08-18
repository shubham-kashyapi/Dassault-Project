Run the file main.py from Anaconda prompt. Pass the data path (directory) and results path (directory) as command line arguments.

Example: python main.py Datasets_csv/Temperature/ Results/

The directory Datasets_csv/Temperature contain csv files (Sheet1.csv, Sheet2.csv, ....) corresponding to Case IDs 1, 2, ...
In a given csv file, every column is treated as a separate time series. Sheet1.csv contains 3 columns: TBLDC, TBLDL, TBLDR 

The segmentation results get saved as plots inside the folder Results/Temperature/Plots/
Paths: For Case ID 1, the images are Results/Temperature/Plots/Sheet1_TBLDC.png, Results/Temperature/Plots/Sheet1_TBLDL.png,
Results/Temperature/Plots/Sheet1_TBLDR.png
For Case ID 2, the images are Results/Temperature/Plots/Sheet2_TBLDC.png and so on.

Similarly, the segmentation results also get saved as json files in the path Results/Temperature/Segmentation_text/
The naming convention is similar to the plots. 
Interpretation of the json files: 
[[[0, 82], "Steady State (S)"], [[83, 123], "Noise (N)"], [[124, 204], "Transition (T)"], ..... ] 
means that the time series is in steady state from time index 0 to 82, noisy from 83 to 123, transition from 124 to 204, and so on.

________________________________________________________________________________________________________________________________________________________________________________

Note (about datasets and results included in this folder):

BLEED F7X Dataset (initial dataset shared with IITM in August 2020):

--Input data paths:   Datasets_csv/Pressure/   [Every csv file in this directory corresponds to a case ID and has 3 columns: PSPCKL, PSPCKOU, PSPCKR]
                      Datasets_csv/Temperature/   [Every csv file in this directory corresponds to a case ID and has 3 columns: TBLDC, TBLDL, TBLDR]
                      Datasets_csv/Altitude/   [Every csv file in this directory corresponds to a case ID and has 1 column: ZP]

--Output paths: (respectively)
  		      Datasets_csv/Pressure/Plots/ and Datasets_csv/Pressure/Segmentation_text/
                      Datasets_csv/Temperature/Plots/ and Datasets_csv/Temperature/Segmentation_text/
                      Datasets_csv/Altitude/Plots/ and Datasets_csv/Altitude/Segmentation_text/             



TS Segmentation Dataset (shared with IITM in March 2021):

--Input data path:   Datasets_csv/TS Segmentation Dataset/ 
  For each distinct flight ID, data from sensors altitude, IASC1A_WAITS1, IASC1B_WAITS1, IASC2A_WAITS2, lbl211b29_11_Temperature_Total_Air is grouped together in a csv file.

--Output paths:
  Datasets_csv/TS Segmentation Dataset/Plots/ and Datasets_csv/TS Segmentation Dataset/Segmentation_text/