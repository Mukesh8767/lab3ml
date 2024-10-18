# -*- coding: utf-8 -*-


Refined version for MultiClass Logistic Regression
"""

## Lab 03 (Part 2): MultiClass Logistic Regression

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

# Load the Iris dataset
data = pd.read_csv('Iris.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Checking unique species in the dataset
print("\nUnique Species in the dataset:")
print(data['Species'].unique())

# Replace species names with numeric labels for multi-class classification
data['Species'] = data['Species'].replace({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3})

# Display updated dataset
print("\nDataset after replacing species names with numeric labels:")
print(data.head())

# Defining features and target
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]  # Features
y = data['Species']  # Target (Species)

# Split the data into training and testing sets (20% training size)
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

# Display the training data
print("\nTraining Data (First 5 rows):")
print(x_train.head())

print("\nTraining Labels (First 5 rows):")
print(y_train.head())

# Initialize the Logistic Regression model
logistic_model = linear_model.LogisticRegression()

# Train the model using the training data
logistic_model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = logistic_model.predict(x_test)

# Display predictions and actual test data
print("\nPredicted Output:")
print(y_pred)

print("\nTest Data (First 5 rows):")
print(x_test.head())

# Calculate and display the model's accuracy score
accuracy = logistic_model.score(x_test, y_test)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix as a heatmap
plt.figure(figsize=(6,4))
sn.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
