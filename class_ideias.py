# import numpy as np ;
from sklearn.linear_model   import LogisticRegression;
import  pandas as  pd;
dataframe =  pd.read_csv('https://sololearn.com/uploads/files/titanic.csv');
# data_x = dataframe[['Age','Fare']].values;
# data_y = dataframe['Survived'].values;
# model = LogisticRegression();
# model.fit(data_x , data_y);
# print(model.coef_ , model.intercept_);


print(dataframe.keys());




from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
