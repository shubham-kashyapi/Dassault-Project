import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
            
class AR_Approach():
    def __init__(self, Ar_Order=3, Mm_Fra=0.1, fill_outlier=False):
        self.Ar_Order=Ar_Order
        self.Mm_Fra=Mm_Fra
        self.fill_outlier=fill_outlier
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.regr = linear_model.LinearRegression()
          
    #%% Preprocessor function
    def Pre_Processor(self, Ser):
        '''
        Works like MATLAB filloutlier function. Drops missing values
        Issue to resolve: Rolling std generates NaN value for the first index, filloutlier performs differently\
            may be due to difference incalculating std, instead of dropping the function should work like \
                MATLAB fillmissing function.
        '''
        Ser.dropna(inplace=True) # Droping missing values
        Ser.index = [i for i in range(len(Ser))]
        if self.fill_outlier==True: # To remove outliers
            Mm_Order=round(len(Ser)*self.Mm_Fra)
            Ser_rollmean=Ser.rolling(Mm_Order+1,min_periods=0).mean()
            Ser_rollstd= Ser.rolling(Mm_Order+1,min_periods=0).std()
            for i in range(len(Ser)):
                if Ser.loc[i]>Ser_rollmean.loc[i]+3*Ser_rollstd.loc[i] \
                    or Ser.loc[i]<Ser_rollmean.loc[i]-3*Ser_rollstd.loc[i]:
                    Ser.loc[i]=np.nan
            Ser.interpolate('nearest', inplace = True)

    #%% Data set function
    def Data_set(self, Ser):
        '''
        Creates datasets X and Y for linear regression
        '''
        self.X.loc[:,0] = Ser.loc[self.Ar_Order-1:0:-1]
        self.X.index=[i for i in range(self.Ar_Order)]
        self.Y.loc[:,0] = Ser.loc[self.Ar_Order:len(Ser)]
        self.Y.index=[i for i in range(len(self.Y))]
        for Order in range(self.Ar_Order+1,len(Ser)):
            self.X.loc[:,Order-self.Ar_Order] = Ser.values[Order-1:Order-self.Ar_Order-1:-1]

    #%% Getting Models
    def Get_Model(self, Ser):
        '''
        Yields coefficints from linear regression from given data sets for given AR order.
        '''
        self.Pre_Processor(Ser)
        self.Data_set(Ser)
        self.regr.fit(self.X.T,self.Y)
        Coef=self.regr.coef_.tolist()
        Inter_Coef=self.regr.intercept_.tolist()
        Inter_Coef.extend(*Coef)
        self.Mod=Inter_Coef # Model
        self.Ypr = self.regr.predict(self.X.T) # Predictions
        self.rsqr=self.regr.score(self.X.T,self.Y)