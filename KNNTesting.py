import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import minmax_scale
from sklearn.pipeline import Pipeline
import time

verbose = False
minkowskiSubdivisions = 5
minkowskiHighExponent = 3 # 2^3 = 8

dataFiles = ["D1heart.csv", "D2heartoutcomes.csv", "D3diabetes.csv", "D4Heart_Disease_Prediction.csv", "D5kidney_disease.csv", "D6kidney_disease.csv", "D7diabetes.csv", "D8Breast_cancer_data.csv"]
distanceMetrics = ['euclidean'] #['euclidean', 'hassanat', 'minkowski', 'chebyshev']
NCA = True
scale = True
OptimizeMeasure = 'accuracy' # 'precision', 'recall', 'accuracy', 'f1score'
OutputFilePath = "/content/drive/MyDrive/MachineLearningProject/KNNOutputNCAprecision.csv"
InputFilePath = "/content/drive/MyDrive/MachineLearningProject/DataSets/"

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
    optimizedMeasureSum = 0
    for dataFile in dataFiles:
        print(dataFile)
        dataset = pd.read_csv(InputFilePath + dataFile, sep=',')
        dataset = dataset.dropna(axis=0, how='any', subset=None, inplace=False)

        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        if scale:
            X = minmax_scale(X, axis=0)

        highAccuracy = 0
        highPrecision = 0
        highRecall = 0
        highf1score = 0
        highK = 0
        for k in range(1, 50, 2):  # 1, 3, 5, 7, 9, ...
            if verbose:
                print("K = ", k)
            if NCA:
                nca = NeighborhoodComponentsAnalysis(max_iter=1000, tol=1e-6)
                model = Pipeline([('nca', nca), ('knn', KNeighborsClassifier(n_neighbors=k, metric=distanceMetric, n_jobs=-1))])
            else:
                if distanceMetric == 'minkowski':
                    model = KNeighborsClassifier(n_neighbors=k, metric=distanceMetric, p=pValue, n_jobs=-1)
                elif distanceMetric == 'hassanat':
                    model = KNeighborsClassifier(n_neighbors=k, metric=hassanat_distance, n_jobs=-1)
                else:
                    model = KNeighborsClassifier(n_neighbors=k, metric=distanceMetric, n_jobs=-1)
            #do multiple scorings
            crossVSAcc = cross_val_score(model, X, y, cv=10, scoring='accuracy')
            crossVSPrecision = cross_val_score(model, X, y, cv=10, scoring='precision')
            crossVSRecall = cross_val_score(model, X, y, cv=10, scoring='recall')
            crossVSF1 = cross_val_score(model, X, y, cv=10, scoring='f1')
            accuracy = crossVSAcc.mean()
            precision = crossVSPrecision.mean()
            recall = crossVSRecall.mean()
            f1score = crossVSF1.mean()
            if (OptimizeMeasure == 'accuracy' and accuracy > highAccuracy) or (OptimizeMeasure == 'precision' and precision > highPrecision) or (OptimizeMeasure == 'recall' and recall > highRecall)or (OptimizeMeasure == 'f1score' and f1score > highf1score):
                highAccuracy = accuracy
                highPrecision = precision
                highRecall = recall
                highf1score = f1score
                highK = k
            if verbose:
                print("accuracy: ", accuracy)
                print("precision: ", precision)
                print("recall: ", recall)
                print("f1score: ", f1score)
        output_file.write(dataFile + "," + str(highK) + "," + str(highAccuracy) + "," + str(highPrecision) + "," + str(highRecall) + "," + str(highf1score) + "\n")
        if OptimizeMeasure == 'accuracy':
            optimizedMeasureSum += highAccuracy
        elif OptimizeMeasure == 'precision':
            optimizedMeasureSum += highPrecision
        elif OptimizeMeasure == 'recall':
            optimizedMeasureSum += highRecall
        elif OptimizeMeasure == 'f1score':
            optimizedMeasureSum += highf1score

    averageOptimizedMetric = optimizedMeasureSum / len(dataFiles)
    output_file.write("Average " + OptimizeMeasure + "," + str(averageOptimizedMetric) + "\n")
    return averageOptimizedMetric


start_time = time.time()
with open(OutputFilePath, 'w') as output_file:
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


    highest_measure = 0
    for key, value in accuraciesDistanceMetricDict.items():
        if value > highest_measure:
            highest_measure = value
            highest_measure_key = key
    output_file.write("Best Distance Formula," + highest_measure_key + "\n")
    output_file.write("Highest Measure," + str(highest_measure) + "\n")
    output_file.write("Time taken: " + str(time.time() - start_time) + "\n")
print("Time taken: ", time.time() - start_time)

