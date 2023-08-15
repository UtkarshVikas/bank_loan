# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load data
data = pd.read_csv('C:/Users/HP/bank.csv')

# Select features (X) and target variable (y)
X = data[['Salary', 'Recurring equal payments']]
y = data['Personal loan']

# Initialize the Linear Regression model
regressor = LinearRegression()

# Fit the model with training data
regressor.fit(X, y)

#Savingf the model to disk
pickle.dump(regressor,open('model.pkl','wb'))

#Loading model to compare the results
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[200000,20000]]))