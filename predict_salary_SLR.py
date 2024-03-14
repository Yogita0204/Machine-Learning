import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\yogay\OneDrive\Desktop\Yogita_Yadav\Data Science\2nd\SIMPLE LINEAR REGRESSION\Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values 

# split the dataset to 80-20%
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m = regressor.coef_ #slope

c = regressor.intercept_ #intercept

y_12 = 9312 * 12 + 26780
y_15 = 9312 * 20 + 26780
y_20 = 9312 * 12 + 26780
y_21 = 9312 * 20 + 26780

bias = regressor.score(X_train, y_train)
bias

variance = regressor.score(X_test,y_test)
variance

