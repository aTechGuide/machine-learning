"""

Reference -> https://medium.com/data-science-everywhere/machine-learning-series-day-5-aeaf0baa103e
"""
import numpy as np


print(np.__version__)

"""
- A NumPy array is simply a collection of the same data typed values.
- NumPy arrays can take two forms, 
  - vectors: Vectors are strictly one dimensional
  - matrices: Matrices are multi-dimensional.
"""
# %%
# create NumPy array from List

np_array = np.array([0,1,2,3,4,5])
print(type(np_array))
print(np_array)

# %%
# Vectorized Operations in NumPy
print(np_array + 2)
print(np_array * 4)
print(np_array / 2)
print(np_array // 2)

# %%
# Creation of array of zeroes/ones
np.zeros((2,2))
np.ones((2,2))

# %%
# Merging of two arrays
arr1 = np.matrix([1,2,3])
arr2 = np.matrix([4,5,6])
print('Horizontal Append:', np.hstack((arr1, arr2)))
print('Vertical Append:', np.vstack((arr1, arr2)))

# %%
# Creating 2 dimensional array from Python list
list1 = [[1,2,3], [4,5,6], [7,8,9]]
arr2d = np.array(list1)
arr2d

# %%
# Creating an array of a particular type
arr2df = np.array(list1, dtype='int') # dtype='float'
arr2df

arr2df.astype('int').astype('str') # astype( ) of NumPy allows us to convert the data type of an array from one type to other.

# %%
list_conv = arr2df.tolist()
print(type(list_conv))

## %% finding the dimension
print('Shape: ', arr2df.shape)
# dtype
print('Datatype: ', arr2df.dtype)
# size
print('Size: ', arr2df.size)
# ndim
print('Num Dimensions: ', arr2df.ndim)

# %%
# Creating a boolean matrix on applying a conditional expression
list3 = [[1, 2, 3, 4],[3, 4, 5, 9], [3, 6, 7, 8]]
arr3 = np.array(list3, dtype='float')
arr3

# %%
boolean_matrix = arr3 > 5
boolean_matrix

# %%
# Reversing an array
arr3[::-1, ]

# %%
#  Inserting nan and inf
arr3[2,1] = np.nan # not a number
arr3[1,2] = np.inf # infinite
arr3

# %%
# Replace nan and inf with -1,
missing_values = np.isnan(arr3) | np.isinf(arr3)
missing_values # <- Returns array of booleans
arr3[missing_values] = -1
arr3

# %%
# Max, min and mean calculations
arr3.mean()
arr3.max()
arr3.min()

# %%
# Row wise and column wise min
print('Column-wise minimum: ', np.amin(arr3, axis=0))
print('Row-wise minimum: ', np.amin(arr3, axis=1))

# %%
# Assigning a portion of an array to other
arr3
arr3a = arr3[:2,:2]
arr3a[:1, :1] = 24 # 24 will be reflected in the parent array too
arr3a
arr3

# %%
#Assign portion of arr3 to arr3_cpy using copy()
arr3
arr3_cpy = arr3[:2,:2].copy()
arr3_cpy[:1, :1] = 1 # 24 will not be reflected in the parent array too
print("Parent aray:",arr3)
arr3_cpy

# %%
# Reshaping arrays
"""
reshape( ) function allows us to change the shape of an array from one form to other.
Note: The reshape() can convert only when rows and columns together must be equal to the number of elements."""
# Reshape a 3x4 array to 2x6 array, here array had 12 elements, hence can be converted to shape (2,6)

arr3.reshape(2,6)

# %%
# Flatten arr3 to a 1d array
arr3.flatten()

# %%
# Changing the flattened array does not change parent
test1 = arr3.flatten()
test1[0] = 1 # changing b1 does not affect arr2

# %%
# Changing the raveled array changes the parent also.
test2 = arr3.ravel()
test2[0] = 0 # changing b2 changes arr2 also

# %%
# Random numbers generation
np.random.rand(3,3)  #Prints random numbers between 0 and 1 in a matrix of order 3x3
np.random.random()
np.random.randint(0, 10, size=[2,2]) # Prints a random number between 0 and 10 in a (2,2) matrix

# %%
# Create random integers of size 10 between [0,10)
np.random.seed(100)
arr_rand = np.random.randint(0, 5, size=10)
arr_rand

# %%
# Unique elements and their count
uniq, count = np.unique(arr_rand, return_counts=True)
uniq
count

# %%
# Pick 15 items from a given list, with equal probability
list4 = np.random.choice(['n','u','m','p','y'], size=15)
uniq1, count1 = np.unique(list4, return_counts=True)
print('Unique items : ', uniq1)
print('Counts : ', count1)

# %%
