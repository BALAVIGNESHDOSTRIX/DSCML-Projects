import numpy as np
from sklearn.linear_model import LinearRegression

pizaa_diameters = np.array([8,12,16,20,24], dtype='<U32').reshape(-1,1)
pizaa_prices = np.array([80,100,120,140,160], dtype='<U32').reshape(-1,1)

model = LinearRegression()
model.fit(pizaa_diameters,pizaa_prices)

user_pizaa_dia = input('Enter the pizaa diameter : ')
prediction = model.predict(np.array([user_pizaa_dia], dtype='float64').reshape(-1,1))
print('Pizaa Price : %.2f ' % prediction)