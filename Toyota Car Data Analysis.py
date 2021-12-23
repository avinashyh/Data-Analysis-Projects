#!/usr/bin/env python
# coding: utf-8

# ## Toyota Car Data Analysis with Pandas

# In[1]:


# Importing libraries
import os
import pandas as pd


# In[2]:


# to load data
cars_data=pd.read_csv('Toyota.csv')


# In[3]:


# to read/view data
cars_data


# In[5]:


# removing extra column from the data frame
cars_data=pd.read_csv('Toyota.csv',index_col=0)

#view modified data frame
cars_data


# In[6]:


# observe junk values in the above data frame like "?","??"
# convert junk value to missing values as Nan
cars_data=pd.read_csv('Toyota.csv',index_col=0,na_values=["??","????","####"])
cars_data


# In[7]:


#creating copy of original Data
cars_data_copy=cars_data.copy()


# In[8]:


cars_data_copy


# In[9]:


# Attributes of Data
# To get the Index(row lables of the data frame)

cars_data_copy.index


# In[10]:


# To get the Columns lables of the data frameb

cars_data_copy.columns


# In[11]:


#to find size
#dataframe.size

cars_data_copy.size


# In[12]:


#to find shape
#dataframe.shape

cars_data_copy.shape


# In[13]:


#to find the memory usage in bytes
#dataframe.memory_usage

cars.memory_usage()


# In[16]:


# to get number of axes/array dimension
#dataframe.ndim : a two dimensional array stores data in the form of rows and columns

cars_data_copy.ndim


# In[18]:


# INdexing and selecting data
# Python slicing operator [] and attribute dot operator '.' are used,
# indexing provides quick and easy access to pandas data structure
# dataframe.head() : returns the first 'n' row from the dataframe 

cars_data_copy.head()
# by default the head() returns first 5 rows


# In[20]:


cars_data_copy.tail()
# by default the tail() returns last 5 rows


# In[21]:


# to access first few rows and last few rows
# at provides lable based scalar lookups

cars_data_copy.at[2,'HP']


# In[23]:


#iat provides integer based lookup

cars_data_copy.iat[5,3]


# In[25]:


# to access group of rows and columns by using Lables using '.loc'

cars_data_copy.loc[:,'HP']


# In[26]:


# CHecking Datatypes of each column
#dataframe.dtypes

cars_data_copy.dtypes


# In[27]:


# Selecting data based on data types 
#dataframe.select_dytpes(include=None,exclude=None)

cars_data_copy.select_dtypes(exclude=[object])


# In[29]:


# Consice Summary of Data Frame
#dataframe.info()

cars_data_copy.info()


# In[31]:


# Getting unique elemts of each column
# unique() is used to find the unique elements of a column 
# Syntax: numpy.unique(array)
# we will umport 'numpy' library

import numpy as np

print(np.unique(cars_data_copy['KM']))


# In[33]:


print(np.unique(cars_data_copy['HP']))


# In[35]:


print(np.unique(cars_data_copy['MetColor']))


# In[37]:


print(np.unique(cars_data_copy['Automatic']))


# In[38]:


print(np.unique(cars_data_copy['Doors']))


# In[40]:


#replace() is used to replace a value with the desired value
#dataframe.replace([to_replace],value,inplace=True)
cars_data_copy.replace('three',3,inplace=True)
cars_data_copy.replace('four',4,inplace=True)
cars_data_copy.replace('five',5,inplace=True)

#COnverting datatype
cars_data_copy['Doors']=cars_data_copy['Doors'].astype('int64')

#view
print(np.unique(cars_data_copy['Doors']))


# In[41]:


cars_data_copy.info()


# In[42]:


#To count the Missing Values in each column
#dataframe.isnull().sum()

cars_data_copy.isnull().sum()


# In[ ]:





# ## Dealing with missing data

# In[43]:


# Subsetting the rows that have one or more missing values

missing=cars_data_copy[cars_data_copy.isnull().any(axis=1)]

missing


# In[44]:


# Two approaches to fill the missing Values
# 1. Fill the missing values by mean/median incase of numerical missing variable
# 2. Fill the missing value with Class which has Maximum count in case of categorical value

cars_data_copy.describe()


# In[45]:


# Imputing missing values of Age

cars_data_copy['Age'].mean()


# In[46]:


# To fill NA/NaN values using the specified value
#DataFrame.fillna()

cars_data_copy['Age'].fillna(cars_data_copy['Age'].mean(),inplace=True)


# In[48]:


# Imputing Missing values of 'KM'

cars_data_copy['KM'].median()


# In[49]:


# To fill NA/NaN values using the specified value
#DataFrame.fillna()

cars_data_copy['KM'].fillna(cars_data_copy['KM'].median(),inplace=True)


# In[50]:


# Imputing missing values of HP

cars_data_copy['HP'].mean()


# In[51]:


# To fill NA/NaN values using the specified value
#DataFrame.fillna()

cars_data_copy['HP'].fillna(cars_data_copy['HP'].mean(),inplace=True)


# In[52]:


# AFter imputin check the data frame

cars_data_copy.isnull().sum()


# In[53]:


# IMputing missing values 'FuelType' (Categorical value/character type)
# Series.value_counts()

cars_data_copy['FuelType'].value_counts()


# In[55]:


# To get the mode value of FuelType

cars_data_copy['FuelType'].value_counts().index[0]


# In[57]:


# To fill na/NaN values 
cars_data_copy['FuelType'].fillna(cars_data_copy['FuelType'].value_counts().index[0],inplace=True)


# In[58]:


# To get the mode value of MetColor

cars_data_copy['MetColor'].mode()


# In[59]:


#To fill the value

cars_data_copy['MetColor'].fillna(cars_data_copy['MetColor'].mode()[0],inplace=True) 


# In[61]:


# Check the values after filling the missing values

cars_data_copy.isnull().sum()


# In[ ]:





# In[ ]:





# ## Exploratory Data Analysis with Python

# In[62]:


# Frequency Tables : to check relationship between the variables, we can do one by one 
# we can check relation between categorical variable using cross tabulation

pd.crosstab(index =cars_data_copy['FuelType'],columns='count', dropna=True)

# we have considered only one categorical variable just to get the frequency distribution


# In[63]:


# Two-Way Tables : IF WE want to check the relationship between two categorical variable
# Here to look at the frequency distribution of 'gearbox types' with repect to different 'fueltype'

pd.crosstab(index=cars_data_copy['Automatic'],columns=cars_data_copy['FuelType'],dropna=True)

# we have looke up at the output interms of numbers


# In[64]:


# Two-Way Table - Join Probability : we can convert the table value into proportion and
# Joint probabilty is the likelihood of two indpenednt happening at the same time

pd.crosstab(index=cars_data_copy['Automatic'], columns=cars_data_copy['FuelType'],normalize=True)


# In[65]:


# Two-Way Table - Marginal Probabilty : probability of the occurance of the single even
# we are basically going to get the row sums and column sums for our table values

pd.crosstab(index=cars_data_copy['Automatic'], columns=cars_data_copy['FuelType'],margins=True,normalize=True)

#Note : probability of cars having manual gearbox when the fuel type are CNG or Diesel 


# In[70]:


#Setting normalize for columns
pd.crosstab(index=cars_data_copy['Automatic'], columns=cars_data_copy['FuelType'],margins=True,normalize=False)


# In[ ]:





# In[71]:


#To compute pairwise correlation of columns excluding NA/NULL values
# Excluding categorical variables to find Pearson's Correlation
# Pearson's correlation is used for to check the strength of association between two numerical value
# If we have ordinal variables then we can go for the other measurea as Kendrall rank c

numerical_data=cars_data_copy.select_dtypes(exclude=[object])

# lets check the no of variables available at under numerical_data

numerical_data.shape


# In[72]:


# Create Correlate matrix

corr_matrix=numerical_data.corr()

corr_matrix


# In[ ]:





# 

# In[ ]:





# ## Data Visualization

# In[73]:


# 1. Scatter Plot : is a set of points that represents the values obtained for two diff
#...(One varies with another) plotted on a horizontal and vertical axis

#When to use Scateer plots
#1. Scatter plots are used convey the relationship between two 'numerical values'
#2. Scatter plots sometimes called correlation plots because they show how to variables

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# importin data
cars_data=pd.read_csv('Toyota.csv',index_col=0,na_values=["??","????"])

#View data
cars_data


# In[74]:


# Removing missing values from the dataframe
cars_data.dropna(axis=0,inplace=True)

# view modified data
cars_data


# In[75]:


# Now plot) Scatter plot
plt.scatter(cars_data['Age'],cars_data['Price'],c='red')
 
# here X = Age, Y= Price and c for couring the points
plt.title("Scatter plot of Price vs Age of the Cars")

#labeliing axis
plt.xlabel("Age(months)")
plt.ylabel("Price(Euros)")
#to show the plot

plt.show()


# In[ ]:



# The above plot shows as the Age increases Price Decreases
# The points are called as markers
# X axis interval called as x-tickcs and similarly y-ticks


# In[76]:


#2. Histogram : is a graphical representation of data using bars of different heights
# Histogram groups number into Ranges and the height of each bar depicts the frequency 
# When to use Histogram
# To represent the frequency distribution of numerical values
#To plot histogram
plt.hist(cars_data['KM'])

# here x=KM and if you just plot it then cant differentiate bin so
plt.hist(cars_data['KM'], color='green', edgecolor='white',bins=5)
plt.title("Histogram of KM of Cars")
plt.xlabel("Kilometers")
plt.ylabel("FRequency")
plt.show()


# In[ ]:


# above plot shows thats frequency distribution KM of the cars shows that most of the cars
#...have travelled between 50000-100000 km and there are only few cars with more distance


# In[77]:


#3. BarPlot:is presents categorical data with rectangular bars with length proportional
# BarPlot:categorical variables and HIstogram:Numerical Variable
# When to use BarPlot
#1. To represent the frequency distribution of categorical variables 
#2. A Bar diagram makes it easy to cpmpare sets of data between different groups

cars_data['FuelType'].value_counts()


# In[79]:


# TO plot BarPlot
counts=[968,116,12]
fueltype=['Petrol','Diesel','CNG']

index=np.arange(len(fueltype))

plt.bar(index,counts,color=['red','green','blue'])
plt.title("Bar Plot for the FuelType")
plt.xlabel("Fuel Types")
plt.ylabel('Frequency')
#plt.xticks(index,fueltyperotation=90)
plt.show()
# index=x, hieght of the bars=counts, 


# In[ ]:





# In[ ]:


# Seaborn Library:
#1. Scatter plot
#2. Histogram plot
#3. Bar plot
#4. Box and Whiskers plot
#5. Pairwise plot
#Seaborn : is a Python data visualization library based on Matplotlib
#--provides high level interface for drawing attractive and informative statistical graphs


# In[80]:


#Scattere plot using Seaborn Library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# importin data
cars_data=pd.read_csv('Toyota.csv',index_col=0,na_values=["??","????"])

#View data
cars_data


# In[81]:


# Removing missing values from the dataframe
cars_data.dropna(axis=0,inplace=True)

# view modified data
cars_data

#observe size of the dataframe


# In[82]:


# Scatter plot "Price vs Age" with default arguements
# set the theme for background

sns.set(style='darkgrid')

#Regression plot(in c bond library)
sns.regplot(x=cars_data['Age'],y=cars_data['Price'])


# In[83]:


# In the above plot there is a line called 'Regression Line'
# By default fit_reg = True
# It estimates and plots a regression model relating the x and y variables 
# what does it mean it basically estimates the coefficient of x and then plots a regres
# why the function is named as regression plot if you dont want reg line then in regplo
# the boxes are called 'Grids' and points are called as 'Markers', you can change the s

# set the theme for background
sns.set(style='darkgrid')

#Regression plot(in c bond library)
sns.regplot(x=cars_data['Age'],y=cars_data['Price'],marker="*")


# In[88]:


#Scatter plot of 'Price vs Age' by 'FuelType'
# by using 'hue' parameter , including other variable to show the fueltype categories 
sns.lmplot(x='Age',y='Price',data=cars_data, fit_reg=False, hue='FuelType', legend=True)

#lmplot : if you want to plot by conditonal subset of data or by including another variable
# legend : which colour represent what category 
# palette : set color palette using the different color palettes that are available.
# we can also customize the appearnace of the markers by using,'Transperancy', 'Shape' etc


# In[89]:


# Histogram plot using Seaborn Library
# Histogram with the default kernel density estimate
sns.histplot(cars_data['Age'])


# Here x="Age"
# distplot :the representation of distribution plot
# Input should be any numerical or any continous variable 
# distplot is about to depreciate


# In[90]:


# BarPlot using Seaborn
# Frequency distribution of fueltype of the cars

sns.countplot(x='FuelType',data=cars_data)


# In[91]:


# Grouped BarPlot
# Grouped Bar plot of 'FuelType' and 'Automatic'

sns.countplot(x='FuelType',data=cars_data, hue='Automatic')


# In[92]:


# Box and Whiskers Plot
# Box and Whiskers Plot - Numerical value
# Box and Whiskers Plot of 'Price' to visually interprete the five-number summary
# The five-number summary include 'mean','median','minimum','Maximum' and 3 quantiles

sns.boxplot(y=cars_data['Price'])


# In[93]:


# Box and Whiskers plot for "Numerical vs Categorical Variable"
# Box and Whisker plot are very useful to check the relationship between 'One Numerical and categorical value'
# Price of the cars for various FuelType

sns.boxplot(x=cars_data['FuelType'],y=cars_data['Price'])


# In[94]:


# Grouped Box and Whiskers plot
# Grouped Box and Whiskers plot of 'Price vs FuelType' and 'Automaic'

sns.boxplot(x='FuelType', y="Price", data=cars_data, hue="Automatic")


# In[95]:


# We can have multiple plots on a window
# Box-Whisker plot and Histogram
# We need to split the plotting window into 2 parts 
# We use subplot() , by giving arguement 2 mean that we need 2 rows

f,(ax_box,ax_hist) =plt.subplots(2,gridspec_kw={"height_ratios":(.25,.75)})
sns.boxplot(cars_data['Price'], ax=ax_box)
sns.distplot(cars_data["Price"],ax=ax_hist, kde=False )


# In[96]:


# Pairwise Plots 
#1. It is used to plot pairwise relationship in a datset 
#2. Create 'scatterplots' for 'joint relationships' and 'histogram' for 'univariate dis
# Pairwise plots used get the plots for all possible combination of variables interms of
# this gives an easier understanding the relationshup between different pairs of variables

sns.pairplot(cars_data, kind="scatter",hue="FuelType")
plt.show()

