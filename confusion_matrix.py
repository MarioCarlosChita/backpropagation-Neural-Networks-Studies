import pandas as pd;
from  sklearn.linear_model   import LogisticRegression ;
# df =  pd.read_csv('https://sololearn.com/uploads/files/titanic.csv');

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics   import  confusion_matrix;
#
# df['male'] = df['Sex'] == 'male'
# X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
# y = df['Survived'].values
# model = LogisticRegression()
# model.fit(X, y)
# y_pred = model.predict(X)


elemento  = [[4,1],
             [3,2]]
model =  confusion_matrix(elemento);

print(model)