import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class AMethod:
    def __init__(self):
        pass
        
    def get_elbow_point(self, xparams, yvals):
        '''
        xparams - 1D numpy array of hyperparameters (length > 2)
        yvals - 1D numpy array of dependent variables for regression (shape same as xparams)
        '''
        self.xparams = MinMaxScaler().fit_transform(xparams.reshape(-1,1))         # Scaling to [0,1]
        self.yvals = MinMaxScaler().fit_transform(yvals.reshape(-1,1)).flatten()   # Scaling to [0,1]
        num_params = self.xparams.shape[0]
        elbow_angle = np.zeros(num_params-2,)
        for elbow_no in range(1, num_params-1):
            xreg1, xreg2 = self.xparams[:elbow_no+1,:], self.xparams[elbow_no:,:]
            yreg1, yreg2 = self.yvals[:elbow_no+1], self.yvals[elbow_no:]
            # Fitting linear models and finding mse
            linmod1 = LinearRegression().fit(xreg1, yreg1)
            linmod2 = LinearRegression().fit(xreg2, yreg2)
            # Computing the angle
            m1, m2 = linmod1.coef_[0], linmod2.coef_[0]
            alpha_angle = np.arctan((m1-m2)/(1+m1*m2))*180/np.pi # Angle in degrees
            alpha_score = (90 - np.abs(alpha_angle))**2
            elbow_angle[elbow_no-1] = alpha_score
   
        opt_elbow = np.argmin(elbow_angle)
        self.opt_idx = opt_elbow + 1
        return self.opt_idx
        
        
    def plot_elbow_curve(self, plot_title, x_variable, y_variable):
        linopt1 = LinearRegression().fit(self.xparams[:self.opt_idx+1].reshape(-1,1), self.yvals[:self.opt_idx+1])
        linopt2 = LinearRegression().fit(self.xparams[self.opt_idx:].reshape(-1,1), self.yvals[self.opt_idx:])
        plt.figure()
        plt.title(str(plot_title + ': Elbow point = {}'.format(self.xparams[self.opt_idx,0])))
        plt.xlabel('{}_scaled'.format(x_variable))
        plt.ylabel('{}_sclaed'.format(y_variable))
        plt.scatter(self.xparams, self.yvals, color = 'b')
        plt.plot(self.xparams[:self.opt_idx+1].reshape(-1, 1), linopt1.predict(self.xparams[:self.opt_idx+1].reshape(-1, 1)), color = 'k')
        plt.plot(self.xparams[self.opt_idx:].reshape(-1, 1), linopt2.predict(self.xparams[self.opt_idx:].reshape(-1, 1)), color = 'k')
        plt.scatter(self.xparams[self.opt_idx], self.yvals[self.opt_idx], color = 'r')
        plt.show()
        return
    
    def write_to_csv(self, output_file_name, x_variable, y_variable):
        xcoords, ycoords, colors, pts_sizes = [], [], [], []
        # Storing points for a scatter plot
        xcoords.extend(list(self.xparams.flatten()))
        ycoords.extend(list(self.yvals))
        colors.extend(self.xparams.shape[0]*['b'])
        pts_sizes.extend(self.xparams.shape[0]*[5])
        # Storing the elbow point
        xcoords.append(self.xparams[self.opt_idx, 0])
        ycoords.append(self.yvals[self.opt_idx])
        colors.append('r')
        pts_sizes.append(5)
        # Storing the left regression line
        linopt1 = LinearRegression().fit(self.xparams[:self.opt_idx+1,:].reshape(-1,1), self.yvals[:self.opt_idx+1])
        xpts_left = np.linspace(self.xparams[0,:], self.xparams[self.opt_idx,:], 1000).flatten()
        xcoords.extend(list(xpts_left))
        ycoords.extend(list(linopt1.predict(xpts_left.reshape(-1,1)).flatten()))
        colors.extend(xpts_left.shape[0]*['k'])
        pts_sizes.extend(xpts_left.shape[0]*[0.5])
        # Storing the right regression line
        linopt2 = LinearRegression().fit(self.xparams[self.opt_idx:,:].reshape(-1,1), self.yvals[self.opt_idx:])
        xpts_right = np.linspace(self.xparams[self.opt_idx,:], self.xparams[-1,:], 1000).flatten()
        xcoords.extend(list(xpts_right))
        ycoords.extend(list(linopt2.predict(xpts_right.reshape(-1,1)).flatten()))
        colors.extend(xpts_right.shape[0]*['k'])
        pts_sizes.extend(xpts_right.shape[0]*[0.5])
        # Storing the points in a csv file
        output_df = pd.DataFrame({'{}_scaled'.format(x_variable): xcoords, '{}_scaled'.format(y_variable): ycoords, 
                                  'Colors': colors, 'Point_sizes': pts_sizes})
        output_df.to_csv('ElbowCurve_{}_{}_vs_{}.csv'.format(output_file_name, y_variable, x_variable), index = False)
        return
        
        