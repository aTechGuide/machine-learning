# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values ## Choosing level
y = dataset.iloc[:, 2:3].values  ## Choosing Salary

# We don't have enough data so we will not be splitting it into test set and training set

# Fitting Random Forest regression to the dataset
from sklearn.ensemble import RandomForestRegressor
rf_regression = RandomForestRegressor(n_estimators = 10, random_state = 0) ## n_estimators tells us how many trees we want to build in a forest

rf_regression.fit(X, y)

# Visualisation of Random Forest results
## We need to Visualise this regression model in high resolution because its non linear and not continuous

X_grid = np.arange(min(X), max(X), .001)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, rf_regression.predict(X_grid), color = 'blue') # Blue line is the prediction of linear regression model
plt.title('Random Forest Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()