"""
Reference 
- https://medium.com/data-science-everywhere/ml-series-day-6-pandas-for-beginners-part-1-4aacad767d1c
"""
#%%
# Importing Pandas
import pandas as pd
print(pd.__version__)
#%%
#Creating an empty dataframe
df1 = pd.DataFrame()
print(df1)

# %%
# From Lists
list2 = ['Peter', 'John', 'Mathew']
df1 = pd.DataFrame(list2)
df1

# %%
# From Dict of Lists
dic_of_list = {'Name':['Peter', 'John', 'Mathew', 'Alex', 'George'],'Age':[12,24,21,34,29]}
df = pd.DataFrame(dic_of_list)
df

# %%
# From Dictionary of Series
name = pd.Series(['Peter', 'John', 'Mathew', 'Alex', 'George'])
age = pd.Series([12,24,21,34,29])

dic2 = {'Name' : name, 'Age' : age}
df3 = pd.DataFrame(dic2)
df3

# %%
# Column Addition
df3['Score'] = pd.Series([54,68,96,71,88])
df3

# %%
# Column Deletion
del df3['Score']
df3
# %%
data = pd.DataFrame([['New Entry',0]],columns = ['Name','Age'])
df3= df3.append(data)
df3
# %%
# Row Deletion
## Rows from a DataFrame is deleted using drop function and corresponding index value.
df3 = df3.drop(4)
df3

# %%
# transpose
## Transposes rows and columns
print(df1)
print(df1.T)

# %%
#axes
## Returns row and column labels
print(df1.axes)

# %%
# ndim
## Returns number of axes/array dimensions.
print(df1.ndim)

# %%
# shape
## Returns dimension of the DataFrame in tuple format with number of rows and columns as values.
print(df1.shape)

# %%
# Sum, Mean, Min, ...
name = pd.Series(['Peter','John','Mathew','Hasan','Alex','Amir'])
age = pd.Series([12,23,16,21,34,24])
score = pd.Series([67,87,56,90,60,85])

df2 = pd.DataFrame({'Name':name, 'Age': age, 'Score':score})
df2

# %%
# Sum
## by default, axis value is 0 [Each column is added individually and strings are appended together.]
print(df2.sum())

print(df2.sum(1)) #[indicates that the axis value is 1]
# %%
# Mean
print(df2.mean()) # Returns the average

# %%
print(df2.std()) # Returns the standard deviation of numerical columns

print(df2.min()) # Returns the min value in each column

print(df2.max()) # Returns the max value in each column

print(df2.count()) # Returns the total number of entries

print(df2.describe()) # Returns the overall information of the numerical columns in a dataframe

# %%
# Indexing and Slicing
## Indexing operator “[ ]” and attribute operator “.” are used to access the specific data.

#Select all rows for a specific column
print(df2.loc[:,'Age'])

# %%
# Select all rows for multiple columns, say list[]
print(df2.loc[:,['Age','Name']])

# %%
# Select few rows for multiple columns, say list[]
print(df2.loc[[0,3,4],['Name','Score']])

# %%
# For getting values with a boolean array
df2
print(df2.loc[3][1]>0) # [row][col]

# %%
# Select all rows for a specific column
print(df2.iloc[:4])

# %%
# Integer slicing
print(df2.iloc[:4])
print(df2.iloc[1:5, 2:4])

# %%
