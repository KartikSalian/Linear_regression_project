import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('Advertising.csv')
print(df.head())

# plotting individual relations
fig,axes=plt.subplots(1,3,figsize=(16,6))
axes[0].plot(df['TV'],df['sales'],'o')
axes[0].set_ylabel('sales')
axes[0].set_title("TV spend")

axes[1].plot(df['radio'],df['sales'],'o')
axes[1].set_ylabel('sales')
axes[1].set_title("Radio spend")

axes[2].plot(df['newspaper'],df['sales'],'o')
axes[2].set_ylabel('sales')
axes[2].set_title("newspaper spend")
plt.tight_layout()

#only features
X= df.drop('sales',axis=1)
print(X)
y=df['sales']

from sklearn.model_selection import train_test_split # this is for breaking data into train and test model

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression #to fit in regression model#
model= LinearRegression()
model.fit(X_train,y_train)
testprediction= model.predict(X_test)
print(testprediction)


from sklearn.metrics import mean_absolute_error,mean_squared_error
mean1=df['sales'].mean()
print(mean1)
sns.histplot(data=df,x='sales',bins=20)


# absolute mean error #
error=mean_absolute_error(y_test,testprediction)
print(error)

#root mean squared error #
error1=mean_squared_error(y_test,testprediction)
print(error1)

error2=np.sqrt(error1)
print(error2)

#Residual plot    #TEST RESIDUALS ie. to check the ideal condition #
test_residual= y_test - testprediction
print(test_residual)

sns.scatterplot(x=y_test,y= test_residual)
plt.axhline(y=0,color='red',ls='--')

sns.displot(test_residual,bins=25,kde=True)

#final model #
final_model= LinearRegression()
final_model.fit(X,y)

print(final_model.coef_)

# prediction graph #
y_hat=final_model.predict(X)
fig,axes=plt.subplots(1,3,figsize=(16,6))
axes[0].plot(df['TV'],df['sales'],'o')
axes[0].plot(df['TV'],y_hat,'o',color='red')
axes[0].set_ylabel('sales')
axes[0].set_xlabel('TV Spend')

axes[1].plot(df['radio'],df['sales'],'o')
axes[1].plot(df['radio'],y_hat,'o',color='red')
axes[1].set_ylabel('sales')
axes[1].set_xlabel('Radio Spend')

axes[2].plot(df['newspaper'],df['sales'],'o')
axes[2].plot(df['newspaper'],y_hat,'o',color='red')
axes[2].set_ylabel('sales')
axes[2].set_xlabel('newspaper Spend')
plt.tight_layout()


from joblib import  dump,load
dump(final_model,'final_sales_model.joblib')


loaded=load('final_sales_model.joblib')
print(loaded.coef_)

#imaginary campaign

#149 tv, 22 radio,12 newspaper

campaign=[[149,22,12]]
result=loaded.predict(campaign)
print(result)

plt.show()