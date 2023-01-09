"""
Created on Thu Jan  5 00:00:44 2023

@author: airron lo
"""
import pandas as pd
import seaborn as sns
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv(r'C:\Users\airro\OneDrive\Desktop\Spyder Projects\kidney_stone.csv')

# counting successes
sns.countplot(x = 'success', data = df)

# coverting the string value (A and B) into binary values
# to apply logistic regression, dropping first column to indicate 
# that we are only using one treatment or the other
Treatment = pd.get_dummies(df['treatment'], drop_first=True)

# checking the first 10 rows of values
Treatment.head(10)

# coverting the string value (large and small) into binary values
Stone_size = pd.get_dummies(df['stone_size'], drop_first=True)

Stone_size.head(10)

# adding in the extra rows to our dataset that we created (Treatment, Stone_size)
df = pd.concat([df, Treatment, Stone_size], axis = 1)

# removing the string columns from the dataset
df.drop(['treatment', 'stone_size'], axis = 1, inplace = True)
df.head(5)

# setting all the other variables as independent
X = df.drop('success', axis = 1)

# value that needs to be predicted, those that survive from heart disease
y = df['success']

# splitting data from testing and training data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# using logistic regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# computing the matrix to check the accuracy of our prediciton from the model
y_pred = log_model.predict(X_test)
confusion_matrix(y_test, y_pred)

# compute accuracy score
accuracy_score(y_test, y_pred)