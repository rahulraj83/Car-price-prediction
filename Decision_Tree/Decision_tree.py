import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

# Load the data from CSV file
df = pd.read_csv('CAR_DETAILS_FROM_CAR_DEKHO.csv')
# Separate features and target
X = df.drop(columns="selling_price")
y = df["selling_price"]

# Encode categorical columns (including 'name')
df_model = df.copy()
label_encoders = {}
for col in ['name', 'fuel', 'seller_type', 'transmission', 'owner']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Features and target
X = df_model.drop('selling_price', axis=1)
y = df_model['selling_price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
model = DecisionTreeRegressor(max_depth=25, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


# print(f"Actual depth of the tree: {model.get_depth()}")

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Plot predicted vs actual prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.tight_layout()
plt.savefig('predicted_vs_actual.png')
plt.show()

importances = model.feature_importances_
features = X.columns

# Visualize
plt.figure(figsize=(8, 6))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.savefig('Feature_Importance.png')
plt.show()
# Function to predict car price
def predict_car_price(name, year, km_driven, fuel, seller_type, transmission, owner):
    input_data = pd.DataFrame([{
        'name': label_encoders['name'].transform([name])[0] if name in label_encoders['name'].classes_ else 0,
        'year': year,
        'km_driven': km_driven,
        'fuel': label_encoders['fuel'].transform([fuel])[0] if fuel in label_encoders['fuel'].classes_ else 0,
        'seller_type': label_encoders['seller_type'].transform([seller_type])[0] if seller_type in label_encoders['seller_type'].classes_ else 0,
        'transmission': label_encoders['transmission'].transform([transmission])[0] if transmission in label_encoders['transmission'].classes_ else 0,
        'owner': label_encoders['owner'].transform([owner])[0] if owner in label_encoders['owner'].classes_ else 0
    }])
    
    prediction = model.predict(input_data)[0]
    return round(prediction, 2)

# Example usage
predicted_price = predict_car_price(
    name='Hyundai Verna 1.6 SX',
    year=2012,
    km_driven=100000,
    fuel='Diesel',
    seller_type='Individual',
    transmission='Manual',
    owner='First Owner'
)

print(f"\nPredicted Selling Price: ₹{predicted_price}")
