import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import minmax_scale

dataFiles = ["D1heart.csv", "D2heartoutcomes.csv", "D3diabetes.csv", "D4Heart_Disease_Prediction.csv", "D7diabetes.csv", "D8Breast_cancer_data.csv"]

for dataFile in dataFiles:
    print(dataFile)
    dataset=pd.read_csv("DataSets/" + dataFile, sep=',')
    dataset = dataset.dropna(axis=0, how='any', subset=None, inplace=False)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

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
