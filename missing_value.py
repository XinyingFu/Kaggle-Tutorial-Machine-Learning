import numpy as np
import pandas as pd
file_path='train.csv'
data=pd.read_csv(file_path)
data=data.select_dtypes(exclude=['object'])
# include only numeric values
test_data=data.copy()
print(data.isnull().sum())
# print missing value by columns
data_nomissing=data.dropna(axis=1)
print(data_nomissing.describe())
# delete columns with missing values
cols_missing=[col for col in data.columns
                                 if data[col].isnull().any()]
reduce_data=data.drop(cols_missing,axis=1)
reduce_test_data=test_data.drop(cols_missing,axis=1)
# delte in the test set colums with missing values in the training set
from sklearn.preprocessing import Imputer
my_imputer=Imputer()
data_imputer=my_imputer.fit_transform(data)
print("Impute Missing Values")
# impute missing values
print("More complex imputation:")
new_data=data.copy()
cols_with_missing = (col for col in new_data.columns
                                    if new_data[col].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()
my_imputer = Imputer()
new_data = my_imputer.fit_transform(new_data)
# note missing Values
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
target=data.SalePrice
predictors=data.drop(['SalePrice'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(predictors,target,train_size=0.7,test_size=0.3,random_state=0)
def score_data(X_train,X_test,y_train,y_test):
    model=RandomForestRegressor()
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    return mean_absolute_error(y_test,preds)
# set up training and test sets, define score
colmissing=[col for col in X_train.columns
                            if X_train[col].isnull().any()]
reduce_X_train=X_train.drop(colmissing,axis=1)
reduce_X_test=X_test.drop(colmissing,axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_data(reduce_X_train,reduce_X_test,y_train,y_test))
# score from dropping columns with missing Values
my_imputer=Imputer()
impute_X_train=my_imputer.fit_transform(X_train)
impute_X_test=my_imputer.transform(X_test)
print("Mean Absolute Error from Imputeration:")
print(score_data(impute_X_train,impute_X_test,y_train,y_test))
# difference between fit_transform and transform?
impute_X_train_plus=X_train.copy()
impute_X_test_plus=X_test.copy()
colmissing=(col for col in X_train.columns
                            if X_train[col].isnull().any())
for col in colmissing:
    impute_X_train_plus[col+'_was_missing']=impute_X_train_plus[col].isnull()
    impute_X_test_plus[col+'_was_missing']=impute_X_test_plus[col].isnull()
my_imputer = Imputer()
impute_X_train_plus = my_imputer.fit_transform(impute_X_train_plus)
impute_X_test_plus = my_imputer.transform(impute_X_test_plus)
print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_data(impute_X_train_plus, impute_X_test_plus, y_train, y_test))
# impute with tracking record
