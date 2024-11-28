import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import minmax_scale

dataset1=pd.read_csv("DataSets/D1heart.csv", sep=',')
dataset1 = dataset1.dropna(axis=0, how='any', subset=None, inplace=False)

X = dataset1.iloc[:, :-1].values
y = dataset1.iloc[:, -1].values

X = minmax_scale(X, axis=0)

for i in range(1, 10, 2): # 1, 3, 5, 7, 9
    print("K = ", i)
    model=KNeighborsClassifier(n_neighbors=i)
    crossVSAcc=cross_val_score(model, X, y, cv=10, scoring='accuracy')
    crossVSPrecision=cross_val_score(model, X, y, cv=10, scoring='precision')
    crossVSRecall=cross_val_score(model, X, y, cv=10, scoring='recall')
    accuracy = crossVSAcc.mean()
    precision = crossVSPrecision.mean()
    recall = crossVSRecall.mean()
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
