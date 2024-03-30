#SUPORT VECTOR REGRESSION

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r'C:\Users\yogay\OneDrive\Desktop\Yogita_Yadav\Data Science\9th\1.POLYNOMIAL REGRESSION\emp_sal.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.svm import SVR
regressor = SVR(kernel='rbf', gamma="auto", degree=5)
regressor.fit(X, y)

y_pred_svr = regressor.predict([[6.5]])

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

