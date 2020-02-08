# Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # Independent Variable
y = dataset.iloc[:, 1].values # Dependent Variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# We don't need feature scaling because the LinearRegression library will take care of it

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression() # creating linearRegressor object
linearRegressor.fit(X_train, y_train) # Fitting model with training set

# Predicting the Test set results
prediction = linearRegressor.predict(X_test)


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, linearRegressor.predict(X_train), color = 'blue')
# Y coordinate is the prediction of train set
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, linearRegressor.predict(X_train), color = 'blue')
# We will obtain same linear regression line by plotting it with either train set or test set
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()