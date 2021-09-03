from sklearn import *
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from matplotlib.pyplot import * 
import matplotlib.pyplot as plt
import numpy as np 
X,Y=datasets.make_regression(n_samples=200,n_features=1,n_targets=1,random_state=0,noise=4.0)
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.3)
reg=linear_model.LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
Coef=reg.coef_
R2=r2_score(Y_test,Y_pred)
MSE=mean_squared_error(Y_test,Y_pred)
print(Coef,R2,MSE)
style.use('ggplot')
plt.scatter(Y_pred,Y_test,color='red')
plt.show()
plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,Y_pred,color='blue',linewidth=2)
plt.show(
