import pandas as pd
from sklearn.model_selection import train_test_split

# Read Data
data = pd.read_csv('train.csv')
print(data.columns)
cols_to_use = ['LotArea', 'MSSubClass', 'OverallQual', 'YrSold', 'YearBuilt']
X = data[cols_to_use]
y = data.SalePrice
train_X, test_X, train_y, test_y = train_test_split(X, y)
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)