# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values ## Choosing level
y = dataset.iloc[:, 2:3].values  ## Choosing Salary

# We don't have enough data so we will not be splitting it into test set and training set

# SVR class doesn't apply Feature Scaling so we need to do it ourself
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
svr_regression = SVR(kernel='rbf')  ## rbf kernel is for non linear problem
## Kernel defines wheather we need linear, polynomial or Gaussian SVR

svr_regression.fit(X, y)

# Visualisation of SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, svr_regression.predict(X), color = 'blue') # Blue line is the prediction of linear regression model
plt.title('SVR Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()