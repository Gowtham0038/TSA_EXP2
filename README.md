# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
# Date: 31/03/2025
# Name: Gowtham C
# Reg.No: 212224240046
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
~~~

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv('/content/powerconsumption.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H:%M')
df = df.sort_values('Datetime')
df['Date_ordinal'] = df['Datetime'].apply(lambda x: x.toordinal())
X = df['Date_ordinal'].values.reshape(-1, 1)
y = df['PowerConsumption_Zone1'].values

~~~
A - LINEAR TREND ESTIMATION
~~~

linear_model = LinearRegression()
linear_model.fit(X, y)
df['Linear_Trend'] = linear_model.predict(X)
plt.figure(figsize=(10, 6))
plt.plot(df['Datetime'], df['PowerConsumption_Zone1'], label='Original Data', color='blue')
plt.plot(df['Datetime'], df['Linear_Trend'], color='yellow', label='Linear Trend')
plt.title('Linear Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Power Consumption Zone 1 (kW)')
plt.legend()
plt.grid(True)
plt.show()

~~~

B- POLYNOMIAL TREND ESTIMATION
~~~

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
df['Polynomial_Trend'] = poly_model.predict(X_poly)
plt.figure(figsize=(10, 6))
plt.plot(df['Datetime'], df['PowerConsumption_Zone1'], label='Original Data', color='blue')
plt.plot(df['Datetime'], df['Polynomial_Trend'], color='green', label='Polynomial Trend (Degree 2)')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Power Consumption Zone 1 (kW)')
plt.legend()
plt.grid(True)
plt.show()

~~~

### OUTPUT
A - LINEAR TREND ESTIMATION
![428668144-88f772c6-ac82-4315-92e3-779e20087708](https://github.com/user-attachments/assets/2b84ef5b-b7af-49e9-9cbf-3087aae16908)

B- POLYNOMIAL TREND ESTIMATION
![428668210-3d92ff4e-de95-438e-b65e-ec006360792a](https://github.com/user-attachments/assets/4ddcae19-6b27-4e73-8740-08da3362f7a1)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
