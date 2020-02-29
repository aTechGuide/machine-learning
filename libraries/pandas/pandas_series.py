"""
Reference 
- https://medium.com/data-science-everywhere/ml-series-day-6-pandas-for-beginners-part-1-4aacad767d1c
"""
#%%

# Importing Pandas
import pandas as pd
print(pd.__version__)

#%%
#cars = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/mtcars.csv')
#Creating an empty Series
s = pd.Series()
s

#%%
# From List
list1 = [12,24,32]
s4 = pd.Series(list1)
s4

#%%

#with custom index values
s5 = pd.Series(list1, index = ['Peter', 'John', 'Mathew'])
s5

#%%
# From NumPy Array
import numpy as np
arr = np.array([12,24,32])
s = pd.Series(arr)
s2 = pd.Series(arr,index = ['Peter', 'John', 'Mathew'])
s
#%%
# From Python Dictionary
dict1 = {'Peter':12, 'John':24, 'Mathew':32}
s3 = pd.Series(dict1)
s3

# %%
#Retrieving data using position
s2[0]

# %%
#retrieve the first three element
s2[:3]

# %%
#retrieve the last three element
s2[-3:]

# %%
# Using label
s2['Peter']
s2[['Peter', 'John', 'Mathew']]

# %%
# axes
## Returns the list of the labels of the series
print(s2.axes)

# %%
# ndim
# Returns dimension of the object.
print(s2.ndim)

# %%
