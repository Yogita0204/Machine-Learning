#build the mlr model & rfe with constat- 42467  

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv(r'C:\Users\yogay\OneDrive\Desktop\Yogita_Yadav\Data Science\3rd,6th\MLR\Investment.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

X=pd.get_dummies(X)

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

X = np.append(arr = np.ones((50,42467)).astype(int), values = X, axis = 1) 

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
