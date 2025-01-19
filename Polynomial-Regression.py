# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset from a CSV file
df = pd.read_csv("house.csv")  # Ensure the file "house.csv" exists in the working directory

# Split the dataset into training and testing sets
msk = np.random.rand(len(df)) < 0.7  # Create a mask for a 70-30 split
train = df[msk]  # Training set
test = df[~msk]  # Testing set

# Extract features (Area) and target (Price) as numpy arrays
x_train = np.asanyarray(train["Area"])  # Training input (Area)
y_train = np.asanyarray(train["Price"])  # Training target (Price)
x_test = np.asanyarray(test["Area"])  # Testing input (Area)
y_test = np.asanyarray(test["Price"])  # Testing target (Price)

# Apply polynomial transformation to the features (degree 2)
poly = PolynomialFeatures(degree=2)
x_train_Poly = poly.fit_transform(x_train.reshape(-1, 1))  # Transform x_train into polynomial features

# Train a linear regression model on the polynomial features
regr = linear_model.LinearRegression()
regr.fit(x_train_Poly, y_train)  # Fit the model to the training data

# Print the coefficients and intercept of the trained model
print(f"Coefficient (linear term): {regr.coef_[1]:.2f}")
print(f"Intercept: {regr.intercept_:.2f}")

# Generate predictions for the range of input values
XX = np.arange(40, 180, 1)  # Range of Area values for prediction
YY = regr.intercept_ + regr.coef_[1] * XX + regr.coef_[2] * XX**2  # Polynomial equation for predictions
y_test_ = regr.intercept_ + regr.coef_[1] * x_test + regr.coef_[2] * x_test**2  # Predictions for test data

# Plot the results
fig = plt.figure()
pic1 = fig.add_subplot(111)
plt.xlabel("Area")  # Label for the x-axis
plt.ylabel("Price")  # Label for the y-axis

# Scatter plot for training data (red points)
pic1.scatter(x_train, y_train, color="red", label="Training Data")
# Scatter plot for testing data (blue points)
pic1.scatter(x_test, y_test, color="blue", label="Testing Data")
# Polynomial regression curve (green line)
pic1.plot(XX, YY, color="green", label="Polynomial Fit")
# Predictions for testing data (yellow points)
pic1.scatter(x_test, y_test_, color="yellow", label="Test Predictions")

# Add legend to the plot
plt.legend()

# Evaluate and print the model's performance metrics
mean_absolute_error = np.mean(np.absolute(y_test - y_test_))  # Mean Absolute Error (MAE)
mse = np.mean((y_test - y_test_)**2)  # Mean Squared Error (MSE)
r2 = r2_score(y_test, y_test_)  # R² Score

print(f"Mean Absolute Error: {mean_absolute_error:.2f}")
print(f"Residual Sum of Squares (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Show the plot
plt.show()
