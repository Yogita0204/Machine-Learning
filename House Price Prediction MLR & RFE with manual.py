#House Price Prediction MLR & RFE with manual

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reading Dataset
dataset=pd.read_csv(r'C:\Users\yogay\OneDrive\Desktop\Yogita_Yadav\Data Science\3rd,6th\MLR\House_data.csv')

#Drop Unnecessary Columns 'id' & 'date' 
dataset = dataset.drop(['id','date'], axis = 1)

#Split Dataframe into Dependent & Independent Variable
X = dataset.iloc[:, 1:]
y = dataset.iloc[:,0]

#Create Dummy Dataset
X=pd.get_dummies(X)

#Splitting Datasets into Training & Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

constant = regressor.intercept_
constant

slope = regressor.coef_
slope

import statsmodels.formula.api as sm  

X = np.append(arr = np.ones((21613,1)).astype(int), values = X, axis = 1)

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,5,6,7,8,9,10,11,12,13,14,15,16,17,18,]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


