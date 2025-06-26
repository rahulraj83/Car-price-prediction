import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

# Load the data from CSV file
df = pd.read_csv('CAR_DETAILS_FROM_CAR_DEKHO.csv')

# Separate features and target
X = df.drop(columns="selling_price")
y = df["selling_price"]

# Convert all categorical columns using OrdinalEncoder
categorical_cols = X.select_dtypes(include="object").columns
encoder = OrdinalEncoder()
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Split data: 60% train, 20% validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Create and train the Decision Tree model on training data
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate on validation data
y_val_pred = model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_r2 = r2_score(y_val, y_val_pred)

# Evaluate on testing data
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

# Print evaluation
print("\nValidation Set Evaluation:")
print(f"Mean Squared Error: {val_mse:.2f}")
print(f"Root Mean Squared Error: {val_rmse:.2f}")
print(f"R² Score: {val_r2:.4f}")

print("\nTest Set Evaluation:")
print(f"Mean Squared Error: {test_mse:.2f}")
print(f"Root Mean Squared Error: {test_rmse:.2f}")
print(f"R² Score: {test_r2:.4f}")

# Plot predicted vs actual prices (test set)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Function to predict car price
def predict_car_price(name, year, km_driven, fuel, seller_type, transmission, owner):
    input_data = pd.DataFrame([{
        "name": name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner
    }])
    input_data[categorical_cols] = encoder.transform(input_data[categorical_cols])
    predicted_price = model.predict(input_data)[0]
    return predicted_price

# Example usage
predicted = predict_car_price(
    name="Renault KWID RXT", 
    year=2016, 
    km_driven=40000, 
    fuel="Petrol", 
    seller_type="Individual", 
    transmission="Manual", 
    owner="First Owner"
)

print("\nPredicted Selling Price:", predicted)
