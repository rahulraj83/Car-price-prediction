import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import tree

# Load the data from CSV file
df = pd.read_csv('CAR_DETAILS_FROM_CAR_DEKHO.csv')

# Display the first few rows and basic information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Feature Engineering
# Extract numerical and categorical features
numeric_features = ['year', 'km_driven']
categorical_features = ['name','fuel', 'seller_type', 'transmission', 'owner']

# Target variable
y = df['selling_price']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create and evaluate the model
X_train, X_test, y_train, y_test = train_test_split(
    df[numeric_features + categorical_features], y, test_size=0.2, random_state=42)

# Create the pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_names = numeric_features + list(model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))
feature_importance = model.named_steps['regressor'].feature_importances_

# Sort features by importance
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Visualize the decision tree (limited to max_depth=3 for better visibility)
limited_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
pipeline_limited = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', limited_tree)
])
pipeline_limited.fit(X_train, y_train)

plt.figure(figsize=(15, 10))
tree.plot_tree(pipeline_limited.named_steps['regressor'], 
               feature_names=feature_names,
               filled=True, 
               rounded=True, 
               fontsize=8)
plt.savefig('decision_tree.png')

# Prediction function
def predict_price(year, km_driven, fuel, seller_type, transmission, owner):
    input_data = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner]
    })
    return model.predict(input_data)[0]

# Example prediction
example_year = 2015
example_km = 50000
example_fuel = 'Petrol'
example_seller = 'Individual'
example_transmission = 'Manual'
example_owner = 'First Owner'

predicted_price = predict_price(
    example_year, example_km, example_fuel, 
    example_seller, example_transmission, example_owner
)

print(f"\nExample Prediction:")
print(f"Car details: {example_year} model, {example_km}km driven, {example_fuel}, {example_seller}, {example_transmission}, {example_owner}")
print(f"Predicted price: ₹{predicted_price:.2f}")

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'regressor__max_depth': [None, 5, 10, 15, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print("\nHyperparameter Tuning:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {-grid_search.best_score_:.2f} (MSE)")

# Final model with best parameters
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
mae_best = mean_absolute_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print("\nTuned Model Evaluation:")
print(f"Mean Squared Error: {mse_best:.2f}")
print(f"Root Mean Squared Error: {rmse_best:.2f}")
print(f"Mean Absolute Error: {mae_best:.2f}")
print(f"R² Score: {r2_best:.4f}")