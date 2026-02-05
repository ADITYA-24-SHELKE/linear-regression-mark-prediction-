import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('studentdata.csv')
print(df.head())
print (df.shape)


print(df.isna().sum())
print(df.duplicated().sum())
print(df.info())
print(df.describe())
print(df.nunique())


print("Categories in 'gender' variable:     ",end=" " )
print(df['gender'].unique())

print("Categories in 'race_ethnicity' variable:  ",end=" ")
print(df['race_ethnicity'].unique())

print("Categories in'parental level of education' variable:",end=" " )
print(df['parental_level_of_education'].unique())

print("Categories in 'lunch' variable:     ",end=" " )
print(df['lunch'].unique())

print("Categories in 'test preparation course' variable:     ",end=" " )
print(df['test_preparation_course'].unique())


numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))



df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average_score'] = df['total_score']/3
print(df.head())


sns.pairplot(df,hue='gender')
plt.show()

import numpy as np
import pandas as pd

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv("studentdata.csv")

# -----------------------------
# 2. Features and target
# -----------------------------
X = data[['reading_score', 'writing_score']]  # Only numeric features
y = data['math_score']

# -----------------------------
# 3. Train-test split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# 4. Train Linear Regression model
# -----------------------------
model = LinearRegression()
model.fit(x_train, y_train)

# -----------------------------
# 5. Predict on test set
# -----------------------------
y_pred = model.predict(x_test)

# -----------------------------
# 6. Evaluate model
# -----------------------------
# MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# R² score
r2 = r2_score(y_test, y_pred)

# Percentage error and accuracy
y_test_array = y_test.to_numpy()
y_pred_array = y_pred

percentage_errors = np.abs((y_test_array - y_pred_array) / np.where(y_test_array == 0, 1, y_test_array)) * 100
accuracy = 100 - np.mean(percentage_errors)

print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)
print(f"Model Accuracy (based on mean percentage error): {accuracy:.2f} %")





