import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('CAR_DETAILS_FROM_CAR_DEKHO.csv')

# Encode categorical variables
label_encoders = {}
categorical_columns = ['name', 'fuel', 'seller_type', 'transmission', 'owner']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for future use

# Normalize numerical features
scaler = MinMaxScaler()
df[['name', 'year', 'km_driven']] = scaler.fit_transform(df[['name', 'year', 'km_driven']])

# Define features (X) and target (y)
X = df.drop(columns=['selling_price'])
y = df['selling_price']

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to numpy arrays
X_train_np = np.c_[np.ones(X_train.shape[0]), X_train.to_numpy()]  # Add bias term
X_test_np = np.c_[np.ones(X_test.shape[0]), X_test.to_numpy()]
y_train_np = y_train.to_numpy().reshape(-1, 1)
y_test_np = y_test.to_numpy().reshape(-1, 1)

# Initialize parameters (theta) with zeros
theta = np.zeros((X_train_np.shape[1], 1))

# Set learning rate and number of iterations
alpha = 0.0001  # Learning rate
iterations = 100000  # Number of iterations

# Implement Gradient Descent

m = len(y_train_np)  # Number of training examples
cost_history = []

for _ in range(iterations):
    predictions = X_train_np @ theta  # Compute predictions
    error = predictions - y_train_np  # Compute error
    gradient = (1 / m) * (X_train_np.T @ error)  # Compute gradient
    theta -= alpha * gradient  # Update theta

    # Compute and store the cost
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    cost_history.append(cost)

# Print the final learned parameters (theta)
print("\nOptimized Theta values:", theta.ravel())
# Model evaluation on test set

y_pred = X_test_np @ theta  # Predictions

# Predictions and evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Plot Actual vs. Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test_np, y_pred, alpha=0.5, color="teal", label="Predicted vs Actual")
plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], color="red", linestyle="--", label="Perfect Fit (y=x)")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.legend()
plt.savefig('predicted_vs_actual.png')
plt.show()



# Function to take user input and predict price

def predict_car_price(name, year, km_driven, fuel, seller_type, transmission, owner):
    # Encode categorical inputs
    try:
        name = label_encoders['name'].transform([name])[0]
        fuel = label_encoders['fuel'].transform([fuel])[0]
        seller_type = label_encoders['seller_type'].transform([seller_type])[0]
        transmission = label_encoders['transmission'].transform([transmission])[0]
        owner = label_encoders['owner'].transform([owner])[0]
    except ValueError:
        print("\n❌ Invalid input! Please enter values exactly as shown in the options.")
        return
    
    user_input_df = pd.DataFrame([[name, year, km_driven]], columns=['name', 'year', 'km_driven'])
    normalized_values = scaler.transform(user_input_df)
    name, year, km_driven = normalized_values[0]
    # Prepare input for prediction
    input_data = np.array([[1, name, year, km_driven, fuel, seller_type, transmission, owner]])
    # Predict price
    predicted_price = input_data @ theta
    print(f"\n💰 Predicted Selling Price of the Car: ₹{predicted_price[0][0]:,.2f}")


predict_car_price(
    name='Maruti 800 AC',
    year=2012,
    km_driven=50000,
    fuel='Petrol',
    seller_type='Individual',
    transmission='Manual',
    owner='First Owner'
)
