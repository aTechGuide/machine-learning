"""
Reference 
- https://medium.com/data-science-everywhere/ml-series-day-6-pandas-for-beginners-part-1-4aacad767d1c
"""

#%%
#Importing Pandas and NumPy
import pandas as pd
import numpy as np

#%%
# Handling missing Data

name = pd.Series(['Peter','John','Mathew','Brick','Alex'])
age = pd.Series([24, np.nan, 18, np.nan, 34])
# Using np.nan we have assigned NaN values manually
df = pd.DataFrame({'Name':name, 'Age':age})

#%%
# Finding Missing Data [Entire Dataframe]
boolDataframe = df.isnull()
# type(boolDataframe)  # pandas.core.frame.DataFrame

# %%

# Finding Missing Data [In Column]
boolSeries = df.Age.isnull() # pandas.core.series.Series
boolSeries.sum()

# %%

# Finding Not Null [ In Column]
boolSeries = df.Age.notnull()
boolSeries.sum()

# %%
# Replace NaN with a Scalar Value
df.fillna(0)

# %%

# Replace NaN with a Mean Value
df = pd.DataFrame({'Name':name, 'Age':age})
df.fillna(round(df.Age.mean(), 0), inplace=True)
df


# %%
# Fill the NaN with the value next to it
df = pd.DataFrame({'Name':name, 'Age':age})
df.fillna(method='backfill')


# %%

# Fill the NaN with the value previous to it
df = pd.DataFrame({'Name':name, 'Age':age})
df.fillna(method='pad')

# %%
# Dropping a missing value
df = pd.DataFrame({'Name':name, 'Age':age})
df.dropna()

# %%

# Replacing a Value
df = pd.DataFrame({'Name':name, 'Age':age})
df.replace({24: 10})

# %%
