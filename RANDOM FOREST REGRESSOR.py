# RANDOM FOREST REGRESSOR

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r'C:\Users\yogay\OneDrive\Desktop\Yogita_Yadav\Data Science\14th\EMP SAL.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor 
reg = RandomForestRegressor(criterion="poisson",n_estimators=50)
reg.fit(X,y)

y_pred = reg.predict([[6.5]])

plt.scatter(X, y, color = 'red')
plt.plot(X,reg.predict(X), color = 'blue')
plt.title('Truth or bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()