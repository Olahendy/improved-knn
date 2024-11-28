import pandas as pd
from sklearn.model_selection import cross_val_score

dataset1=pd.read_csv("DataSets/D1heart.csv", sep=',')
dataset1 = dataset1.dropna(axis=0, how='any', subset=None, inplace=False)