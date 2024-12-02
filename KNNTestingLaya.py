import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
import time

verbose = False
minkowskiSubdivisions = 5
minkowskiHighExponent = 3  # 2^3 = 8

dataFiles = [
    "D1heart.csv", "D2heartoutcomes.csv", "D3diabetes.csv",
    "D4Heart_Disease_Prediction.csv", "D5kidney_disease.csv",
    "D6kidney_disease.csv", "D7diabetes.csv", "D8Breast_cancer_data.csv"
]
distanceMetrics = ['hassanat']  # ['euclidean', 'hassanat', 'minkowski', 'chebyshev']
NCA = False
scale = False
OptimizeMeasures = ['accuracy', 'precision', 'recall', 'f1score']  # Dynamic measures
OutputFilePath = "./content"
InputFilePath = "/Users/slaya/improved-knn/DataSets/"


# This function implements the Hassanat distance formula
def hassanat_distance(df1, df2):
    dist_list = []
    for x in range(len(df1)):
        data1 = np.array(df1)[x]
        data2 = np.array(df2)[x]

        min_ = min(data1, data2)
        max_ = max(data1, data2)

        if min_ >= 0:
            dist = 1 - ((1 + min_) / (1 + max_))
        else:
            dist = 1 - ((1 + min_ + np.abs(min_)) / (1 + max_ + np.abs(min_)))
        dist_list.append(dist)
    return np.sum(dist_list)


def evaluate_knn_for_metric(distanceMetric, pValue, dataFiles, output_file, optimizeMeasure):
    optimizedMeasureSum = 0
    for dataFile in dataFiles:
        print(dataFile)
        dataset = pd.read_csv(InputFilePath + dataFile, sep=',')
        dataset = dataset.dropna(axis=0, how='any', subset=None, inplace=False)

        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        if scale:
            X = minmax_scale(X, axis=0)

        highMetric = 0
        highK = 0
        for k in range(1, 50, 2):  # 1, 3, 5, ...
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

            # Perform cross-validation with the specific scoring metric
            scoring_function = make_scorer(globals()[f"{optimizeMeasure}_score"])
            cross_val_scores = cross_val_score(model, X, y, cv=10, scoring=scoring_function)
            metric_mean = cross_val_scores.mean()

            if metric_mean > highMetric:
                highMetric = metric_mean
                highK = k

            if verbose:
                print(f"{optimizeMeasure}: {metric_mean}")

        # Write results for this dataset
        output_file.write(dataFile + "," + str(highK) + "," + str(highMetric) + "\n")
        optimizedMeasureSum += highMetric

    # Compute average metric value
    averageOptimizedMetric = optimizedMeasureSum / len(dataFiles)
    output_file.write("Average " + optimizeMeasure + "," + str(averageOptimizedMetric) + "\n")
    return averageOptimizedMetric


start_time = time.time()
with open(OutputFilePath, 'w') as output_file:
    accuraciesDistanceMetricDict = dict()
    for optimizeMeasure in OptimizeMeasures:  # Loop through each measure dynamically
        output_file.write(f"Optimizing for {optimizeMeasure}\n")
        for distanceMetric in distanceMetrics:
            if distanceMetric == 'minkowski':
                for i in range(0, minkowskiHighExponent * minkowskiSubdivisions + 1):
                    pValue = 2 ** (i / minkowskiSubdivisions)
                    output_file.write(f"Distance Metric,{distanceMetric},p={pValue}\n")
                    averageMetric = evaluate_knn_for_metric(distanceMetric, pValue, dataFiles, output_file, optimizeMeasure)
                    accuraciesDistanceMetricDict[distanceMetric + ", " + str(pValue)] = averageMetric
            else:
                output_file.write(f"Distance Metric,{distanceMetric}\n")
                averageMetric = evaluate_knn_for_metric(distanceMetric, None, dataFiles, output_file, optimizeMeasure)
                accuraciesDistanceMetricDict[distanceMetric] = averageMetric

        # Find the highest metric value for the current optimization measure
        highest_metric = 0
        for key, value in accuraciesDistanceMetricDict.items():
            if value > highest_metric:
                highest_metric = value
                highest_metric_key = key
        output_file.write(f"Best Distance Formula for {optimizeMeasure}," + highest_metric_key + "\n")
        output_file.write(f"Highest {optimizeMeasure}," + str(highest_metric) + "\n")

output_file.write("Time taken: " + str(time.time() - start_time) + "\n")
print("Time taken: ", time.time() - start_time)
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import pairwise_distances
# from sklearn.preprocessing import minmax_scale
# import matplotlib.pyplot as plt
# import logging
# import time
# import os

# # Configuration
# K_VALUES = range(1, 50, 2)  # Odd values for K
# DATA_FILES = ["DataSets/D1heart.csv", "DataSets/D2heartoutcomes.csv", "DataSets/D3diabetes.csv", "DataSets/D4Heart_Disease_Prediction.csv",
#               "DataSets/D5kidney_disease.csv", "DataSets/D6kidney_disease.csv", "DataSets/D7diabetes.csv", "DataSets/D8Breast_cancer_data.csv"]
# DISTANCE_METRICS = ['hassanat']  # Extendable: ['euclidean', 'minkowski', 'chebyshev']
# SCALE = False
# OUTPUT_FOLDER = "./output/"  # Folder to save CSVs and images
# OPTIMIZE_MEASURES = ['accuracy', 'recall']  # List of metrics to optimize for

# # Ensure output folder exists
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # Logging configuration
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# # Hassanat distance function
# def hassanat_distance(x, y):
#     """
#     Compute the Hassanat distance between two points x and y.
#     """
#     min_ = np.minimum(x, y)
#     max_ = np.maximum(x, y)

#     if np.all(min_ >= 0):
#         return 1 - (1 + min_).sum() / (1 + max_).sum()
#     else:
#         return 1 - ((1 + min_ + abs(min_)).sum() / (1 + max_ + abs(min_)).sum())

# def calculate_cross_val_scores(model, X, y):
#     """
#     Calculate cross-validation scores for various metrics.
#     """
#     scores = {
#         'accuracy': cross_val_score(model, X, y, cv=10, scoring='accuracy').mean(),
#         'precision': cross_val_score(model, X, y, cv=10, scoring='precision').mean(),
#         'recall': cross_val_score(model, X, y, cv=10, scoring='recall').mean(),
#         'f1score': cross_val_score(model, X, y, cv=10, scoring='f1').mean(),
#     }
#     return scores

# def evaluate_knn_for_metric(distanceMetric, dataFile):
#     """
#     Evaluate KNN for a specific distance metric for one dataset.
#     """
#     dataset = pd.read_csv(dataFile)
#     dataset.dropna(axis=0, inplace=True)

#     X = dataset.iloc[:, :-1].values
#     y = dataset.iloc[:, -1].values

#     if SCALE:
#         X = minmax_scale(X, axis=0)

#     results = []
#     for k in K_VALUES:
#         if distanceMetric == "hassanat":
#             model = KNeighborsClassifier(
#                 n_neighbors=k, 
#                 metric="pyfunc", 
#                 metric_params={"func": hassanat_distance}
#             )
#         else:
#             model = KNeighborsClassifier(n_neighbors=k, metric=distanceMetric)

#         scores = calculate_cross_val_scores(model, X, y)
#         results.append({'K': k, **scores})

#     return results

# def save_results_to_csv(results, dataFile, optimize_measure):
#     """
#     Save evaluation results to a CSV file.
#     """
#     base_name = os.path.basename(dataFile).replace('.csv', '')
#     output_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_{optimize_measure}_results.csv")
#     pd.DataFrame(results).to_csv(output_file, index=False)
#     logging.info(f"Results for {optimize_measure} saved to {output_file}")

# def plot_results(metrics_dict, dataFile, optimize_measure):
#     """
#     Plot results for distance metrics comparison and save as PNG.
#     """
#     base_name = os.path.basename(dataFile).replace('.csv', '')
#     output_image = os.path.join(OUTPUT_FOLDER, f"{base_name}_{optimize_measure}_comparison.png")

#     plt.bar(metrics_dict.keys(), metrics_dict.values())
#     plt.xlabel("K Value")
#     plt.ylabel(f"Average {optimize_measure.capitalize()}")
#     plt.title(f"KNN Results for {base_name} ({optimize_measure.capitalize()})")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig(output_image)
#     plt.close()
#     logging.info(f"Visualization for {optimize_measure} saved to {output_image}")

# if __name__ == "__main__":
#     start_time = time.time()
#     for dataFile in DATA_FILES:
#         results = evaluate_knn_for_metric(DISTANCE_METRICS[0], dataFile)
#         for optimize_measure in OPTIMIZE_MEASURES:
#             # Save results for the current optimization measure
#             save_results_to_csv(results, dataFile, optimize_measure)

#             # Find the best K for this optimization measure
#             best_result = max(results, key=lambda x: x[optimize_measure])
#             logging.info(f"Best K for {dataFile} ({optimize_measure}): {best_result['K']} with {optimize_measure} = {best_result[optimize_measure]:.4f}")

#                         # Prepare data for plotting
#             metrics_dict = {result['K']: result[optimize_measure] for result in results}
#             plot_results(metrics_dict, dataFile, optimize_measure)

#     logging.info(f"Total time taken: {time.time() - start_time:.2f} seconds")