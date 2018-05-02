import pandas as pd
file_path='train.csv'
data=pd.read_csv(file_path)
# import data
print(data.describe())
print(data.columns)
# check data features and feature names
price_data=data.SalePrice
print(price_data.head())
# pick one feature and see the first few values
columns_interest=['LotArea','YearBuilt']
interest_data=data[columns_interest]
print(interest_data.describe())
# pick several features together and see the value distribution
y=data.SalePrice
# put what you want to predict into y
predictors=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X=data[predictors]
# put all useful variable to predict y into X
from sklearn.tree import DecisionTreeRegressor
model_DTR=DecisionTreeRegressor()
model_DTR.fit(X,y)
# use decisiontree to fit the model
n=10
print("Making prediciton for the following", n, "houses:")
print(X.head(n))
print("The predictions are:")
print(model_DTR.predict(X.head(n)))
# make predictions using the fitted model
from sklearn import linear_model
model_LR=linear_model.LinearRegression()
model_LR.fit(X,y)
# use linear regresson to fit the model
n=10
print("Making prediciton for the following", n, "houses:")
print(X.head(n))
print("The predictions are:")
print(model_LR.predict(X.head(n)))
# make predictions using the fitted model_DTR
from sklearn.metrics import mean_absolute_error
predict_price=model_DTR.predict(X)
print(mean_absolute_error(y,predict_price))
# check mean absolute error of the decisiontree model
predict_price=model_LR.predict(X)
print(mean_absolute_error(y,predict_price))
# check mean absolute error of the linear regression model
from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=0)
model=DecisionTreeRegressor()
model.fit(train_X,train_y)
val_prediction=model.predict(val_X)
print(mean_absolute_error(val_y,val_prediction))
# split sample into training set and validation set.
# fit the model using the training set.
# find the mean absolute error of the fitted model on the validation set
