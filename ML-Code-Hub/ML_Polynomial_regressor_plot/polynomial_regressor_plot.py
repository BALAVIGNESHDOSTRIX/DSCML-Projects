import pandas as pd 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Load the data points
data_frame = pd.read_csv('gold.csv',error_bad_lines=False)

#Column Slicing
x = data_frame.iloc[:,0:1].values
y = data_frame.iloc[:,1].values

#Initial The Polynomial Features
poly=PolynomialFeatures(degree=2)
poly_x=poly.fit_transform(x)


regressor=LinearRegression()
regressor.fit(poly_x,y)
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(poly_x),color='blue')
plt.show()