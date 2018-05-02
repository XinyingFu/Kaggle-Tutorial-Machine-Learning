import pandas as pd
file_path='train.csv'
data=pd.read_csv(file_path)
print(data.describe())
print(data.columns)
price_data=data.SalePrice
print(price_data.head())
