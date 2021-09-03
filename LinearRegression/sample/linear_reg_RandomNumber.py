import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score

#declaring the random X_train and Y_train data_sets
x_train = array([1,2,3,4,5,6,7,8,9,10])
y_train= array([0.2,0.5,0.8,0.9,1.3,1.7,2,2.5,3,3.1])

#declaring the random x_test and y_test data_sets
x_test= array([11,12])
y_test= array([3.3,3.5])

#Converting the 1D into 2D show we are using numpy reshape() method for X_train,Y_train,x_test,y_test
x_train=x_train.reshape(x_train.shape[0],1)
y_train=y_train.reshape(y_train.shape[0],1)
x_test=x_test.reshape(x_test.shape[0],1)
y_test=y_test.reshape(y_test.shape[0],1)

#creating the Variable for LinearRegression Model
regr=linear_model.LinearRegression()

#finding the coefficient for X_train,Y_train
regr.fit(x_train,y_train)

#finding the prediction using test dataset
y_prediction=regr.predict(x_test)

#printing the accuracy of the test data_sets
accuracy =regr.score(x_test,y_test)
print("accuracy:" + str(float(accuracy)))

print("coefficient:"+ str(float(regr.coef_)))
print('MSE :'+ str(float(mean_squared_error(y_test,y_prediction))))
print('Variance: '+str(float(r2_score(y_test,y_prediction))))

#Plot the data points on the graph using matlib
x=np.concatenate((x_train,x_test),axis=0)
y=np.concatenate((y_train,y_test),axis=0)
y_p=regr.predict(x)
plt.scatter(x,y, color='black')
plt.plot(x,y_p,color='green',linewidth=2)
plt.show()
