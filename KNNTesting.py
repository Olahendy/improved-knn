import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import minmax_scale

verbose = True

dataFiles = ["D1heart.csv", "D2heartoutcomes.csv", "D3diabetes.csv", "D4Heart_Disease_Prediction.csv", "D5kidney_disease.csv", "D6kidney_disease.csv", "D7diabetes.csv", "D8Breast_cancer_data.csv"]

with open("KNNOutput.csv", 'w') as output_file:
    distanceMetrics = ['euclidean', 'minkowski', 'manhattan', 'chebyshev']
    accuraciesPerDistanceMetric = []
    for distanceMetric in distanceMetrics:
        output_file.write("Distance Metric," + distanceMetric + "\n")
        accuracySum = 0
        
        for dataFile in dataFiles:
            print(dataFile)
            dataset=pd.read_csv("DataSets/" + dataFile, sep=',')
            dataset = dataset.dropna(axis=0, how='any', subset=None, inplace=False)

            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values

            X = minmax_scale(X, axis=0)
            highAccuracy = 0
            highPrecision = 0
            highRecall = 0
            highK = 0
            for k in range(1, 10, 2): # 1, 3, 5, 7, 9
                if verbose:
                    print("K = ", k)
                model=KNeighborsClassifier(n_neighbors=k, metric=distanceMetric)
                crossVSAcc=cross_val_score(model, X, y, cv=10, scoring='accuracy')
                crossVSPrecision=cross_val_score(model, X, y, cv=10, scoring='precision')
                crossVSRecall=cross_val_score(model, X, y, cv=10, scoring='recall')
                accuracy = crossVSAcc.mean()
                precision = crossVSPrecision.mean()
                recall = crossVSRecall.mean()
                if accuracy > highAccuracy:
                    highAccuracy = accuracy
                    highPrecision = precision
                    highRecall = recall
                    highK = k
                if verbose:
                    print("accuracy: ", accuracy)
                    print("precision: ", precision)
                    print("recall: ", recall)
            output_file.write(dataFile + "," + str(highK) + "," + str(highAccuracy) + "," + str(highPrecision) + "," + str(highRecall) + "\n")
            accuracySum += highAccuracy
        averageAccuracy = accuracySum/len(dataFiles)
        output_file.write("Average Accuracy," + str(averageAccuracy) + "\n")
        accuraciesPerDistanceMetric.append(averageAccuracy)
    highest_accuracy = 0
    for i in range(len(accuraciesPerDistanceMetric)):
        if accuraciesPerDistanceMetric[i] > highest_accuracy:
            highest_accuracy = accuraciesPerDistanceMetric[i]
            highest_accuracy_index = i
    output_file.write("Best Distance Formula," + str(distanceMetrics[highest_accuracy_index]) + "\n")
    output_file.write("Highest Accuracy," + str(highest_accuracy) + "\n")