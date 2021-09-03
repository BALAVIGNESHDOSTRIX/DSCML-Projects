import numpy as np 
import pandas as pd 
from matplotlib.pyplot import *
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
from sklearn import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
df=pd.read_csv('Advertising.csv')
# print(df.head())
TV=df.ix[:,1]
Radio=df.ix[:,2]
Newspaper=df.ix[:,3]
Sales=df.ix[:,4]
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(TV,Radio,Sales)
plt.show()
X=pd.concat([TV,Radio,Newspaper],axis=1)
Y=Sales
# print(X.head())
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.3)
reg=linear_model.LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
Coef=reg.coef_
R2=r2_score(Y_test,Y_pred)
MSE=mean_squared_error(Y_test,Y_pred)
print(Coef,R2,MSE)
style.use('ggplot')
plt.scatter(Y_pred,Y_test)
plt.title('Predicted values vs Real values')
plt.xlabel('Y_pred')
plt.ylabel('Y_test')
plt.show()
fig2=plt.figure()
ax2=Axes3D(fig2)
ax2.scatter(X_test['TV'],X_test['Radio'],Y_test,color='red')
ax2.scatter(X_test['TV'],X_test['Radio'],Y_pred,color='blue')
plt.show()
