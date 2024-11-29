import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import minmax_scale
import time

verbose = False
minkowskiSubdivisions = 5
minkowskiHighExponent = 3 # 2^3 = 8

dataFiles = ["D1heart.csv", "D2heartoutcomes.csv", "D3diabetes.csv", "D4Heart_Disease_Prediction.csv", "D5kidney_disease.csv", "D6kidney_disease.csv", "D7diabetes.csv", "D8Breast_cancer_data.csv"]

def evaluate_knn_for_metric(distanceMetric, pValue, dataFiles, output_file):
    accuracySum = 0
    for dataFile in dataFiles:
        print(dataFile)
        dataset = pd.read_csv("DataSets/" + dataFile, sep=',')
        dataset = dataset.dropna(axis=0, how='any', subset=None, inplace=False)

        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        X = minmax_scale(X, axis=0)
        highAccuracy = 0
        highPrecision = 0
        highRecall = 0
        highK = 0
        for k in range(1, 10, 2):  # 1, 3, 5, 7, 9
            if verbose:
                print("K = ", k)
            if distanceMetric == 'minkowski':
                model = KNeighborsClassifier(n_neighbors=k, metric=distanceMetric, p=pValue)
            else:
                model = KNeighborsClassifier(n_neighbors=k, metric=distanceMetric)
            crossVSAcc = cross_val_score(model, X, y, cv=10, scoring='accuracy')
            crossVSPrecision = cross_val_score(model, X, y, cv=10, scoring='precision')
            crossVSRecall = cross_val_score(model, X, y, cv=10, scoring='recall')
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
    averageAccuracy = accuracySum / len(dataFiles)
    output_file.write("Average Accuracy," + str(averageAccuracy) + "\n")
    return averageAccuracy


start_time = time.time()
with open("KNNOutput.csv", 'w') as output_file:
    distanceMetrics = ['minkowski', 'chebyshev']
    accuraciesDistanceMetricDict = dict()
    for distanceMetric in distanceMetrics:
        if(distanceMetric == 'minkowski'):
            for i in range(0,minkowskiHighExponent*minkowskiSubdivisions + 1):
                pValue = 2**(i/minkowskiSubdivisions)
                output_file.write("Distance Metric," + distanceMetric + ",p=" + str(pValue) + "\n")
                if pValue == 0:
                    output_file.write("Also called Euclidean distance\n")
                if pValue == 1:
                    output_file.write("Also called Manhattan distance\n")
                

                averageAccuracy = evaluate_knn_for_metric(distanceMetric, pValue, dataFiles, output_file)
                accuraciesDistanceMetricDict[distanceMetric + ", " + str(pValue)] = averageAccuracy
        else:
            output_file.write("Distance Metric," + distanceMetric + "\n")
            averageAccuracy = evaluate_knn_for_metric(distanceMetric, None, dataFiles, output_file)
            accuraciesDistanceMetricDict[distanceMetric] = averageAccuracy
            
        
    highest_accuracy = 0
    for key, value in accuraciesDistanceMetricDict.items():
        if value > highest_accuracy:
            highest_accuracy = value
            highest_accuracy_key = key
    output_file.write("Best Distance Formula," + highest_accuracy_key + "\n")
    output_file.write("Highest Accuracy," + str(highest_accuracy) + "\n")
print("Time taken: ", time.time() - start_time)
