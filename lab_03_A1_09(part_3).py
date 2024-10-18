# -*- coding: utf-8 -*-
"""Lab_03_A3_44(Part 3).ipynb

Refined version for Lab 03 (Part 3): Understanding Underfitting and Overfitting
"""

## Lab 03 (Part 3): Underfitting and Overfitting in Data

# Import necessary libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model

# Load the dataset (Age vs Employment status)
data = pd.read_csv('Age.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Scatter plot of Age vs Employed
plt.figure(figsize=(8,6))
plt.xlabel('Age')
plt.ylabel('Employed (Y/N)')
plt.title('Age vs Employment Status')
plt.scatter(data['Age'], data['Employed'], marker='*')
plt.show()

# Linear Regression Model
linmodel = linear_model.LinearRegression()
linmodel.fit(data[['Age']], data['Employed'])

# Plotting the Linear Regression fit
plt.figure(figsize=(8,6))
plt.xlabel('Age')
plt.ylabel('Employed (Y/N)')
plt.title('Linear Regression: Age vs Employment Status')
plt.scatter(data['Age'], data['Employed'], marker='*', label='Data Points')
plt.plot(data['Age'], linmodel.predict(data[['Age']]), color='green', label='Linear Fit')
plt.legend()
plt.show()

# Logistic Regression Model
logmodel = linear_model.LogisticRegression()
logmodel.fit(data[['Age']], data['Employed'])

# Plotting the Logistic Regression fit
plt.figure(figsize=(8,6))
plt.xlabel('Age')
plt.ylabel('Employed (Y/N)')
plt.title('Logistic Regression: Age vs Employment Status')
plt.scatter(data['Age'], data['Employed'], marker='*', label='Data Points')
plt.plot(data['Age'], logmodel.predict(data[['Age']]), color='green', label='Logistic Fit')
plt.legend()
plt.show()

# Evaluate Linear Regression model
lin_score = linmodel.score(data[['Age']], data['Employed'])
print(f"Linear Regression Model Score: {lin_score:.2f}")

# Evaluate Logistic Regression model
log_score = logmodel.score(data[['Age']], data['Employed'])
print(f"Logistic Regression Model Score: {log_score:.2f}")
