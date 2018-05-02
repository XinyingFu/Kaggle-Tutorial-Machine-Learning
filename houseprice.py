import pandas as pd
file_path='train.csv'
data=pd.read_csv(file_path)
print(data.describe())
print(data.columns)
price_data=data.SalePrice
print(price_data.head())
columns_interest=['LotArea','YearBuilt']
interest_data=data[columns_interest]
print(interest_data.describe())
y=data.SalePrice
predictors=['LotArea','YearBuilt','BedroomAbvGr','YrSold','KitchenAbvGr','PoolArea']
X=data[predictors]
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model.fit(X,y)
print("Making prediciton for the following 5 houses:")
print(X.head())
print("The predictions are:")
print(model.predict(X.head()))
