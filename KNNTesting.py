import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import minmax_scale
import time

verbose = False
minkowskiSubdivisions = 5
minkowskiHighExponent = 3 # 2^3 = 8

dataFiles = ["D1heart.csv", "D2heartoutcomes.csv", "D3diabetes.csv", "D4Heart_Disease_Prediction.csv", "D5kidney_disease.csv", "D6kidney_disease.csv", "D7diabetes.csv", "D8Breast_cancer_data.csv"]
distanceMetrics = ['euclidean'] #['minkowski', 'chebyshev']
NCA = True

#This function is taken from someone else's hassanat distance formula here: https://www.kaggle.com/code/banddaniel/hassanat-distance-implementation-w-knn
def hassanat_distance(df1, df2):
    dist_list = []
    total = 0
    
    for x in range(len(df1)):
        data1 = np.array(df1)[x]
        data2 = np.array(df2)[x]
        
        min_ = min(data1, data2)
        max_ = max(data1, data2)
        
        if min_ >= 0:
            dist = 1-( (1+min_)/(1+max_) )
            dist_list.append(dist)

        else:
            dist = 1-( (1+min_ + np.abs(min_))/(1+max_+np.abs(min_) ) )
            dist_list.append(dist)
    
    total = np.sum(dist_list)
    return total

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
            if NCA:
                nca = NeighborhoodComponentsAnalysis()
                nca.fit(X, y)
                X = nca.transform(X)
            if verbose:
                print("K = ", k)
            if distanceMetric == 'minkowski':
                model = KNeighborsClassifier(n_neighbors=k, metric=distanceMetric, p=pValue, n_jobs=-1)
            elif distanceMetric == 'hassanat':
                model = KNeighborsClassifier(n_neighbors=k, metric=hassanat_distance, n_jobs=-1)
            else:
                model = KNeighborsClassifier(n_neighbors=k, metric=distanceMetric, n_jobs=-1)
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
    output_file.write("Time taken: " + str(time.time() - start_time) + "\n")
print("Time taken: ", time.time() - start_time)

