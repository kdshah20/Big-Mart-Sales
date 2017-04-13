# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:40:12 2017

@author: Karan
"""

import pandas as pd

train = pd.read_csv('C:/Users/Karan/BigMartSales/Train.csv')

train.dtypes

'''Univariate Analysis'''

# Get mean, median, mode, range of continuous variables
train.describe()

#Get categorical variables
categorical_variables = train.dtypes.loc[train.dtypes == 'object'].index
print(categorical_variables)

#Get number of unique values for each categorical variable(column)
train[categorical_variables].apply(lambda x:len(x.unique()))


#print the counts of each category
train['Item_Fat_Content'].value_counts()
train['Outlet_Type'].value_counts()

'''Multivariate Analysis'''

#Categorical and Categorical Variables

#in this case we look at cross tabulation of two variables.

ct = pd.crosstab(train['Outlet_Type'],train['Outlet_Size'],margins = True)
%matplotlib inline
ct.iloc[:-1,:-1].plot(kind='bar',stacked=True, color=['red','blue','green'], grid=False)

'''Missing Value Treatment'''

#Outlet size and Item_Weight have missing values.
#From my previous analysis item_weight I think Item_Weight is not a significant feature
#so may be we can drop it before we fit our model to the classifier

#We can treat Outlet_Size for missing values

from scipy.stats import mode

mode(train['Outlet_Size'].tolist()).mode[0]

#Fill the missing values of Outlet_Size using mode
train['Outlet_Size'].fillna(mode(train['Outlet_Size'].tolist()).mode[0],inplace=True)

# check that there are no missing values now in Outlet_Size
train.apply(lambda x: sum(x.isnull()))

'''Variable Transformation'''

#Item_Fat_Content contains multiple category names referring to the same category
#We can reduce the number of categories by combining categories referring to the 
#same amount of fat content

FatCont = train['Item_Fat_Content']
for i in range(len(FatCont)):
    if FatCont[i] == 'LF' or FatCont[i] == 'low fat':
        FatCont[i] = 'Low Fat'
    elif FatCont[i] == 'reg':
        FatCont[i] = 'Regular'

# check that our categories have reduced
train['Item_Fat_Content'].value_counts()/train.shape[0]
              

#From our multivariate analysis we know that both Supermarket Type 2 and Type 3 have
#the same size. We can combine these together
train.Outlet_Type.value_counts()

categories_to_combine = ['Supermarket Type2','Supermarket Type3']

for cat in categories_to_combine:
    train['Outlet_Type'].replace({cat:'Others'},inplace=True)
    
#check new categories in train['Outlet_Type']
train['Outlet_Type'].value_counts()/train.shape[0]

#Check for missing values
train.apply(lambda x: sum(x.isnull()))


train['Item_Type'].value_counts()/train.shape[0]
