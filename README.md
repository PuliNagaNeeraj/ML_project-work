# ML_project-work

## AIM :
To Develop an accurate and reliable weather prediction model that takes into account multiple meteorological variables, historical weather data, and advanced machine learning algorithms to provide precise forecasts for various geographic locations and time horizons.

## EQUIPMENTS REQUIRED :
Hardware PC's

Jupyter Notebook with python installation

## PROGRAM :
```
Developed By : PULI NAGA NEERAJ
Register Number : 212223240130
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset

weather_df = pd.read_csv("weather.csv")

# Data preprocessing

weather_df.dropna(inplace=True)  # Drop missing values

# Feature engineering: Adding month and day as features

weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df['month'] = weather_df['date'].dt.month
weather_df['day'] = weather_df['date'].dt.day
X = weather_df.drop(columns=['date', 'weather'])
y = weather_df['weather']

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)

# Feature scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training

model = RandomForestClassifier(n_estimators=100, random_state=101)
model.fit(X_train, y_train)

# Predictions

predictions = model.predict(X_test)

# Evaluation

print(classification_report(y_test, predictions))
print(f'Accuracy: {accuracy_score(y_test, predictions):.2f}')

# Example prediction

test_data = {
    'precipitation': 10.9,
    'temp_max': 10.6,
    'temp_min': 2.8,
    'wind': 4.5,
    'month': 5,  # Example month
    'day': 17    # Example day
}

test_df = pd.DataFrame([test_data])
test_df_scaled = scaler.transform(test_df)
print(model.predict(test_df_scaled))
```
# OUTPUT :
![image](https://github.com/23004426/ML_Workshop/assets/144979327/4aebdf5a-5b33-4f26-b6fe-eb5caabb14cc)


# RESULT :
Thus the weather prediction model is developed ,trained and tested for various geographic locations and time horizons successfully.
