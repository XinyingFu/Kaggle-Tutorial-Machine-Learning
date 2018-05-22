import pandas as pd
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
file_path='train.csv'
for i in data.columns:
    print(data[i].head(5))
def get_some_data():
    data=pd.read_csv(file_path)   
    print(data.describe())
    print(data.columns)
    print(data.head())
    y=data['SalePrice']
    cols_to_use=['MSSubClass','LotArea','YearBuilt']
    X=data[cols_to_use]
    my_imputer=Imputer()
    imputed_X=my_imputer.fit_transform(X)
    return imputed_X, y
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
X,y=get_some_data()
my_model = GradientBoostingRegressor()
my_model.fit(X, y)
my_plots = plot_partial_dependence(my_model,       
                                   features=[0,1,2], # column numbers of plots we want to show
                                   X=X,            # raw predictors data.
                                   feature_names=['MSSubClass','LotArea','YearBuilt'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis
