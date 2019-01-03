# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values ## Choosing level
y = dataset.iloc[:, 2].values  ## Choosing Salary

# We don't have enough data so we will not be splitting it into test set and training set


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()
linearRegressor.fit(X,y) # We have also created a linear model to compare its efficiency with polynomial model


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyRegression = PolynomialFeatures(degree=3)

X_poly = polyRegression.fit_transform(X) ## Creating new metrics X_poly with polynomial feature

lin_reg_2 = LinearRegression() 
lin_reg_2.fit(X_poly, y) # Creating a polynomial model with polinomial metrics X_poly

# Visualisation of Linear Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, linearRegressor.predict(X), color = 'blue') # Blue line is the prediction of linear regression model
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualisation of Polynomial Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(polyRegression.fit_transform(X)), color = 'blue') # Blue line is the prediction of linear regression model
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()