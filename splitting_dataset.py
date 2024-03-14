import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r'C:\Users\yogay\OneDrive\Desktop\Yogita_Yadav\Data Science\Data.csv')

X = dataset.iloc[:, :-1].values	
y = dataset.iloc[:,3].values  

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="median") 

imputer = imputer.fit(X[:,1:3]) 
X[:, 1:3] = imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X[:,0]) 
X[:,0] = labelencoder_X.fit_transform(X[:,0])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#splitting 80% 20%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2,random_state=0) 
#splitting 70% 30%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3,random_state=0) 
#splitting 60% 40%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.4,random_state=0) 
#splitting 75% 25%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.25,random_state=0) 

#set random state!=0
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2) 

#set random state=0
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2,random_state=0) 

