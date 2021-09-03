import numpy as np 
import pandas as pd 
df=pd.read_csv('GlobalTemperatures.csv')
year=df.ix[1800:,0]
# Landmax temp
X1=df.ix[1800:,3]
#  Landmin temp
X2=df.ix[1800:,5]
#Avg Land Temp
Y=df.ix[1800:,1]
# print(Y.head())
#Show data
from matplotlib.pyplot import *
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(X1,X2,Y)
plt.show()
X=pd.concat([X1,X2],axis=1)
# print(X.head())
from sklearn import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
reg=linear_model.LinearRegression()
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.25)
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
Coef=reg.coef_
R2=r2_score(Y_test,Y_pred)
MSE=mean_squared_error(Y_test,Y_pred)
print(Coef,R2,MSE)
style.use('ggplot')
plt.scatter(Y_pred,Y_test,color='blue')
plt.show()
fig2=plt.figure()
ax2=Axes3D(fig2)
ax2.scatter(X_test['LandMaxTemperature'],X_test['LandMinTemperature'],Y_test,color='red')
ax2.scatter(X_test['LandMaxTemperature'],X_test['LandMinTemperature'],Y_pred,color='green')
plt.show()
