# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:00:46 2025

@author: hp
"""
#importing libraries
import pandas as pd
import matplotlib.pyplot as plt




#reading data from files
data = pd.read_csv("advertising.csv")


#to visualize data
fig , axs = plt.subplots(1,3,sharey = True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])


#creating x&y for linear regression
feature_cols = ['TV']
X = data[feature_cols ]
y = data.Sales


#importing linear regression algorithm
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)



result = 6.97+0.055*50
print(result)


#create a dataframe with min and max value of the table
X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()


preds = lr.predict(X_new)
preds


data.plot(kind = 'scatter',x='TV',y='Sales')

plt.plot(X_new,preds,c='red',linewidth = 3)

import statsmodels.formula.api as smf
lr = smf.ols(formula = 'Sales ~ TV',data = data).fit()
lr.conf_int()


#finding the probability values
lr.pvalues


#Finding the R-Squared values
lr.rsquared


#Multi linear Regression
feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
y = data.Sales


lr = LinearRegression()
lr.fit(X, y)


print(lr.intercept_)
print(lr.coef_)


lr = smf.ols(formula = 'Sales ~ TV+Radio+Newspaper',data = data).fit()
lr.conf_int()
lr.summary()



lr = smf.ols(formula = 'Sales ~ TV+Radio',data = data).fit()
lr.conf_int()
lr.summary()