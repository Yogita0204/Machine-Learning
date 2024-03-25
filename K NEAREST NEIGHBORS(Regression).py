# K NEAREST NEIGHBORS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r'C:\Users\yogay\OneDrive\Desktop\Yogita_Yadav\Data Science\9th\1.POLYNOMIAL REGRESSION\emp_sal.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.neighbors import KNeighborsRegressor
regressor_knn = KNeighborsRegressor(n_neighbors=6, weights="uniform", algorithm="auto")
regressor_knn.fit(X,y)

y_pred_knn = regressor_knn.predict([[6.5]])

plt.scatter(X, y, color = 'red')
plt.plot(X,  regressor_knn.predict(X), color = 'blue')
plt.title('Truth or Bluff (KNN)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
