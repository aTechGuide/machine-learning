"""
Reference 
- https://medium.com/data-science-everywhere/ml-series-day-6-pandas-for-beginners-part-1-4aacad767d1c
"""

"""
GroupBy performs
- split
- apply
- combine

operations on data

It splits the object, applies the function and combines the result.

The basic split-apply-combine operation can be computed with the groupby() method passing the name of the desired key column of the DataFrames. 
It will result in DataFrameGroupByobject. 

"""

#%%
#Importing Pandas and NumPy
import pandas as pd
import numpy as np

# %%

data = {
  'Name': ['Peter', 'John', 'Kerome', 'Mathew', 'Paul', 'Niel', 'Orton', 'Whisk', 'Alex', 'Hasan'],
  'Age': [12, 22, 62, 23, 33, 46, 19, 51, 29, 45],
  'Category': ['Young','Young','Adult','Mid','Mid','Adult','Young','Adult','Young','Adult'],
  'Total_Score':[86,79,63,83,41,82,56,78,64,71]
  }

df = pd.DataFrame(data)

# %%

dfGroupBy = df.groupby('Category') # pandas.core.groupby.generic.DataFrameGroupBy
dfGroupBy.groups

# %%
# Grouping based on multiple columns
df.groupby(['Category','Total_Score']).groups

# %%
#Iterating through each group

for name, group in dfGroupBy:
  print(f"name = {name}\n")
  print(group)

# %%
#  select a single group.
dfGroupBy.get_group('Young')

# %%
