"""
Reference 
- https://medium.com/data-science-everywhere/ml-series-day-6-pandas-for-beginners-part-1-4aacad767d1c
"""

#%%
#Importing Pandas and NumPy
import pandas as pd
import numpy as np

# %%
df = pd.DataFrame(np.random.randn(5,4) ,columns = list(['Sample1','Sample2','Sample3','Sample4']))
df

# %%
# Plotting Graphs
df.plot()

# %%
df.plot.bar()


# %%
# Generates bar graph with stacks
df.plot.bar(stacked=True)


# %%

# To generate a horizontally stacked bar graph
df.plot.barh(stacked=True)

# %%
# Histograms can be plotted using the plot.hist() method. 
# We should specify the number of bins in which the data will be grouped.
df.plot.hist(bins=10)

# %%
# To generate box plots
df.plot.box()

# %%
# To generate scatter plots,
df.plot.scatter(x='Sample1', y='Sample2')

# %%
