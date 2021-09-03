from numpy import *
from matplotlib.pyplot import *
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn import *
diabetes=datasets.load_diabetes()
# print(diabetes.keys()) 
X=diabetes.data[:,np.newaxis,3]
Y=diabetes.target
# print(Y.shape)
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.2)
# print(X_train.sha?pe)
reg=linear_model.LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
Coef=reg.coef_
print(Coef)
R2=r2_score(Y_test,Y_pred)
MSE=mean_squared_error(Y_test,Y_pred)
print(R2,MSE)
style.use('ggplot')
plt.scatter(Y_pred,Y_test,color='blue')
plt.title('Predicted vs Real values')
plt.xlabel('Y_pred')
plt.ylabel('Y_test')
plt.show()
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,Y_pred,color='blue',linewidth=2)
plt.show()
