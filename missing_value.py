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
