import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale

dataset1=pd.read_csv("DataSets/D1heart.csv", sep=',')
dataset1 = dataset1.dropna(axis=0, how='any', subset=None, inplace=False)

X = dataset1.iloc[:, :-1].values
y = dataset1.iloc[:, -1].values

