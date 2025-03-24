# %%

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


# %% [markdown]
# ### Load the Data for Modeling and Predictions

# %%

# Load the datasets
train_data = pd.read_csv('train.csv')
weather_data = pd.read_csv('Weather.csv')
test_data = pd.read_csv('test.csv')

# Convert datetime formats
train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'], format='%b %d, %Y %I%p')
weather_data['date_time'] = pd.to_datetime(weather_data['date_time'])
test_data['Timestamp'] = pd.to_datetime(test_data['Timestamp'], format='%b %d, %Y %I%p')


# %% [markdown]
# # Data Exploration and Visualization for Solar Energy Prediction

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets with handling of bad lines
metadata = pd.read_csv('metadata.csv', encoding='ISO-8859-1', on_bad_lines='skip')
sample_submission = pd.read_csv('sample_submission.csv', encoding='ISO-8859-1', on_bad_lines='skip')
solar_irradiance_2014 = pd.read_csv('Solar_Irradiance_2014.csv', encoding='ISO-8859-1', on_bad_lines='skip')
solar_irradiance_2015 = pd.read_csv('Solar_Irradiance_2015.csv', encoding='ISO-8859-1', on_bad_lines='skip')
solar_irradiance_2016 = pd.read_csv('Solar_Irradiance_2016.csv', encoding='ISO-8859-1', on_bad_lines='skip')
solar_irradiance_2017 = pd.read_csv('Solar_Irradiance_2017.csv', encoding='ISO-8859-1', on_bad_lines='skip')
test = pd.read_csv('test.csv', encoding='ISO-8859-1', on_bad_lines='skip')
train = pd.read_csv('train.csv', encoding='ISO-8859-1', on_bad_lines='skip')
weather = pd.read_csv('Weather.csv', encoding='ISO-8859-1', on_bad_lines='skip')

# Display first few rows of each dataset
metadata.head(), sample_submission.head(), solar_irradiance_2014.head(), train.head(), weather.head()

# %% [markdown]
# ### Merging Weather Data with Solar Production Data

# %%
# Convert datetime formats and merge
weather['date_time'] = pd.to_datetime(weather['date_time'])
train['Timestamp'] = pd.to_datetime(train['Timestamp'], format='%b %d, %Y %I%p')

# Merge datasets based on timestamps
merged_data = pd.merge(train, weather, how='inner', left_on='Timestamp', right_on='date_time')

# Dropping unnecessary columns
merged_data = merged_data.drop(columns=['date_time', 'moon_illumination', 'moonrise', 'moonset', 'sunrise', 'sunset', 'DewPointC'])

# Show first few rows of the merged dataset
merged_data.head()

# %% [markdown]
# ### Correlation Analysis

# %%
# Calculate correlation matrix
correlation_matrix = merged_data.corr()

# Sort correlation with '% Baseline'
correlation_with_baseline = correlation_matrix['% Baseline'].sort_values(ascending=False)

# Display correlations
correlation_with_baseline

# %% [markdown]
# ### Data Visualization - Distribution of Weather Variables

# %%
# Plotting histograms for weather-related variables
weather_variables = ['tempC', 'humidity', 'sunHour', 'precipMM', 'cloudcover', 'windspeedKmph']

plt.figure(figsize=(12, 8))
for i, variable in enumerate(weather_variables):
    plt.subplot(2, 3, i+1)
    sns.histplot(merged_data[variable], bins=20, kde=True)
    plt.title(f'Distribution of {variable}')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Correlation Heatmap

# %%
# Heatmap for correlation between all variables
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Solar Production and Weather Factors')
plt.show()

# %% [markdown]
# ### Scatter Plot between Solar Production and Key Weather Factors

# %%
# Scatter plots for % Baseline vs. key weather factors
key_factors = ['tempC', 'humidity', 'sunHour', 'cloudcover']

plt.figure(figsize=(12, 8))
for i, factor in enumerate(key_factors):
    plt.subplot(2, 2, i+1)
    sns.scatterplot(x=merged_data[factor], y=merged_data['% Baseline'])
    plt.title(f'% Baseline vs. {factor}')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Data Exploration - Checking for Missing Values and Outliers

# %%
# Checking for missing values
missing_values = merged_data.isnull().sum()

# Summary statistics for outlier detection
summary_statistics = merged_data.describe()

# Display missing values and summary statistics
missing_values, summary_statistics

# %% [markdown]
# # Solar Energy Prediction using Random Forest

# %%

# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare dataset for modeling by selecting features and target
features = ['tempC', 'humidity', 'sunHour', 'precipMM', 'cloudcover', 'windspeedKmph']
target = '% Baseline'

# Drop rows with missing values for simplicity in this step
merged_data_clean = merged_data.dropna(subset=features + [target])

# Define the features (X) and the target (y)
X = merged_data_clean[features]
y = merged_data_clean[target]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = rf_model.predict(X_val)

# Calculate RMSE for model evaluation
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {rmse}')


# %%
# Initialize and train the RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = rf_model.predict(X_val)

# Calculate RMSE for model evaluation
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {rmse}')

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Define the parameter grid
rf_params = {
  'n_estimators': [100, 200, 300, 400, 500],  # Number of trees in the forest
  'max_depth': [2, 3, 4, 5],  # Maximum depth of each tree
  'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
  'min_samples_leaf': [1, 5, 10],  # Minimum samples required at each leaf node
  'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
  'bootstrap': [True, False]
}

# Create a RandomForestRegressor model
rf_model = RandomForestRegressor()

# Perform Randomized Search CV
random_search = RandomizedSearchCV(
  estimator=rf_model,
  param_distributions=rf_params,
  scoring='neg_mean_squared_error',
  cv=5,
  n_iter=100,
  verbose=2
)

random_search.fit(X_train, y_train)

# Get the best model and parameters
best_model = random_search.best_estimator_
best_params = random_search.best_params_

print("Best Hyperparameters:", best_params)

# Use the best_model for prediction on new data

# %%
best_params={'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_features': 'log2', 'max_depth': 5, 'bootstrap': False}

# %%
# Initialize and train the RandomForestRegressor
rf_model = RandomForestRegressor(**best_params)
rf_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = rf_model.predict(X_val)

# Calculate RMSE for model evaluation
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {rmse}')

# %% [markdown]
# ### Predictions on Test Data

# %%

# Ensure both Timestamp and date_time are in datetime format
test_data['Timestamp'] = pd.to_datetime(test_data['Timestamp'], format='%b %d, %Y %I%p')
weather_data['date_time'] = pd.to_datetime(weather_data['date_time'])

# Merge test data with weather data
test_data_clean = pd.merge(test_data, weather_data, how='left', left_on='Timestamp', right_on='date_time')
test_data_clean = test_data_clean.drop(columns=['date_time', 'moonrise', 'moonset', 'sunrise', 'sunset'])

# Handle missing values in test data by dropping rows with NaN
test_data_clean = test_data_clean.dropna(subset=features)

# Make predictions on the test set
test_predictions = rf_model.predict(test_data_clean[features])

# Prepare submission output in the format of sample_submission
output = test_data[['Timestamp']].copy()
output['% Baseline'] = test_predictions

# Save the predictions to a CSV file
output.to_csv('solar_energy_predictions.csv', index=False)
print("Predictions saved to 'solar_energy_predictions.csv'")



