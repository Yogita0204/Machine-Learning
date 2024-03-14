import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r'C:\Users\yogay\OneDrive\Desktop\Yogita_Yadav\Data Science\1.POLYNOMIAL REGRESSION\emp_sal.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# linear regression visualizaton 
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# poly nomial visualization 
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color = 'purple')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'yellow')
plt.title('Truth or Bluff (Polynomial Regression)--degree-6')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color = 'green')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'red')
plt.title('Truth or Bluff (Polynomial Regression)--degree-3')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'purple')
plt.title('Truth or Bluff (Polynomial Regression)--degree-4')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color = 'black')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'red')
plt.title('Truth or Bluff (Polynomial Regression)--degree-5')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


