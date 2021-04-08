import streamlit as st
import sys
import numpy as np
import pandas as pd
from Interval_Halving import Modified_Interval_Halving
from optimal_hyperparameter import AMethod
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import time
from is_Anomalous import is_Anomalous

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_hi('Loading streamline UI')
    st.title('Anamoly Detection from Time Series Data')

    df = pd.DataFrame({
        'first column': ['Interval Halving', 'Auto Regression']
    })

    st.sidebar.title('Input Options')

    processing_option = st.sidebar.radio("Process Single time series or Batch", ('Single', 'Batch'))

    if processing_option == 'Single':
        st.write('Please input file containing Single time series')
    else:
        st.write("Please input file containing the multiple time series data")

    option = st.sidebar.selectbox(
        'Select the Model',
        df['first column'])
    'Algorithm Selected: ', option



    uploaded_file = st.file_uploader("Select a file(csv) containing Time Series Data")
    if uploaded_file is not None:
        st.header('Input')
        st.write(uploaded_file.name)

        df_data = pd.read_csv(uploaded_file)
        st.line_chart(df_data)
        if st.checkbox('Show Input Data'):
            st.write(df_data)
#        'size of the dataset: ',
#        st.write(df_data.size)

#       Preprocess data / input validation  (for df_data)

        if (option == "Interval Halving") :

            for colname in list(df_data.columns):

                tseries = df_data[colname].dropna().to_numpy()
                err_series = np.var(tseries)
                # Hyperparameter tuning
                var_fracs = np.linspace(0.01, 0.1, 20)
                count_intervals = np.zeros(20, )
                my_bar = st.progress(0)
                for frac_ind, frac in enumerate(var_fracs):
                    time.sleep(0.1)
                    my_bar.progress(frac*10)
                    model = Modified_Interval_Halving(noise_err=frac * err_series, min_window_len=100, \
                                                      title=colname, suppress_print=True)
                    count_intervals[frac_ind] = model.train_model(tseries)

                elbow_model = AMethod(var_fracs, count_intervals)
                elbow_idx = elbow_model.get_elbow_point()
                elbow_model.write_to_csv(colname, 'Variance_fraction', 'Number_of_intervals')
                opt_var_fraction = var_fracs[elbow_idx]
                # Running interval halving
                final_model = Modified_Interval_Halving(noise_err=opt_var_fraction * err_series, min_window_len=100, \
                                                        ylabel=colname, suppress_print=True)
                final_model.train_model(tseries)
                final_model.write_to_csv(colname)

                pspckl_elbow = pd.read_csv('ElbowCurve_PSPCKL.csv')
                pspckl_elbow

                st.header('Output')
                fig = plt.Figure(figsize=(15,12))
                ax1 = plt.subplot(111)
                #            ax2 = plt.subplot(313)

                fig = ax1.scatter(pspckl_elbow['Variance_fraction'], pspckl_elbow['Number_of_intervals'], \
                                  color=pspckl_elbow['Colors'])
                #            ax1.title('Elbow Curve')
                plt.xlabel(pspckl_elbow.columns[0])
                plt.ylabel(pspckl_elbow.columns[1])
                #plt.show()
                plt.title('Elbow Curve')
                st.pyplot(plt)

                pspckl_interval = pd.read_csv('IntervalHalving_PSPCKL.csv')
                pspckl_interval

                fig2 = plt.Figure(figsize=(15, 12))
                ax2 = plt.subplot(211)

                ax2.scatter(pspckl_interval['Time (sec)'], pspckl_interval['PSPCKL'], \
                            color=pspckl_interval['Colors'], s=0.5)
                plt.title('Regimes of Operation')
                #            ax2.title('Elbow Curve')
                plt.xlabel(pspckl_interval.columns[0])
                plt.ylabel(pspckl_interval.columns[1])
                #            ax2.xlabel(pspckl_interval.columns[0])
                #            ax2.ylabel(pspckl_interval.columns[1])
                st.pyplot(plt)

        elif (option == "Auto Regression"):

            from sklearn import linear_model
            Type = 'PSPCKL'
            Var_Data = df_data
            AR_order = 3

            Var_Data = pd.Series(Var_Data.iloc[:, 0])
            Mod_Avg = {'PSPCKL': [0.03498, 1.1563, -0.3082, 0.1394], 'PSPCKR': [0.0339, 1.1617, -0.3140, 0.1401], \
                       'PSPCKOU': [0.0103, 1.0744, -0.2272, 0.1428], 'TBLDC': [0.0251, 1.0588, 0.1254, -0.1845], \
                       'TBLDL': [0.1638, 1.4501, -0.2972, -0.1536], 'TBLDR': [0.1382, 1.4448, -0.2928, -0.1527]}
            Comp_With = np.array(Mod_Avg[Type])

            # Setting up data for Regression Model
            X = pd.DataFrame({'0': Var_Data.loc[AR_order - 1:0:-1]})  # replace with AR_order-1
            Y = pd.DataFrame({Var_Data[AR_order]})
            for order in range(AR_order + 1, len(Var_Data)):
                X.loc[:, order - AR_order] = Var_Data.values[order - 1:order - AR_order - 1:-1]
                Y[order - AR_order] = Var_Data[order]

            # Setting up regression model and obtaining coefficients
            regr = linear_model.LinearRegression()
            regr.fit(X.T, Y.T)
            Model_Arr = np.array([regr.intercept_[0], regr.coef_[0, 0], regr.coef_[0, 1], regr.coef_[0, 2]])

            # Calculate Angle
            Mod_Unit = Model_Arr / np.linalg.norm(Model_Arr)
            Mod_Avg_Unit = Comp_With / np.linalg.norm(Comp_With)
            Theta = np.rad2deg(np.arccos(np.dot(Mod_Unit, Mod_Avg_Unit.T)))

            # Verdict
            if Theta <= 15:
                Verdict = 'The data represents NORMAL operation'
            else:
                Verdict = 'The data represents ANOMALOUS operation'

            # Return Angle, AR coefficients, Verdict
            print('The AR coefficients for order {} are:'.format(AR_order))
            print('\nIntercept: {0:.3f} \nCoef_1: {1:.3f} \nCoef_2: {2:.3f} \nCoef_3: {3:.3f}'.format( \
                regr.intercept_[0], regr.coef_[0, 0], regr.coef_[0, 1], regr.coef_[0, 2]))
            print('\nAngle: {0:.3f} Degree \n'.format(Theta))

            st.header('Output')
            st.text('Theta for the data set:{:.3f}.'.format(Theta))
            st.text('The AR coefficients for the model are:\n [{:.3f}, {:.3f}, {:.3f}, {:.3f}]'.format(regr.intercept_[0],\
                                                                                     regr.coef_[0, 0], regr.coef_[0, 1], regr.coef_[0, 2]), )

            theta_file = st.file_uploader("Select a file(csv) with comparison data")
            if theta_file is not None:
                theta_data = pd.read_csv(theta_file)

                if st.checkbox('Show Comparison Data'):
                    st.write(theta_data)

                theta_data['Colors'] = 'b'
                theta_data['size'] = 3

                theta_data.loc[len(theta_data.index)] = [2,Theta, 'r', 10]
                fig3 = plt.Figure(figsize=(15, 12))
                ax3 = plt.subplot(111)

                scat = ax3.scatter(theta_data['Case ID'], theta_data['Theta'],  \
                                color=theta_data['Colors'], s=theta_data['size'])

                plt.title('Theta Value - Comparison Plot')
                plt.xlabel(theta_data.columns[0])
                plt.ylabel(theta_data.columns[1])
                plt.grid()
                plt.minorticks_on()
                st.pyplot(plt)







