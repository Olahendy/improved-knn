import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


outputTableOfAccuracies = False #Do we want to output all 10 cross validation scores for each model?


dataset=pd.read_csv("DataSets/drug200-1.csv", sep=',')
dataset = dataset.dropna(axis=0, how='any', subset=None, inplace=False)

dummies = pd.get_dummies(dataset.BP, prefix='BP')
merged = pd.concat([dataset, dummies], axis='columns')
merged = merged.drop(['BP'], axis='columns')

dummies = pd.get_dummies(dataset.Cholesterol, prefix='Cholesterol')
merged = pd.concat([merged, dummies], axis='columns')
merged = merged.drop(['Cholesterol'], axis='columns')

dummies = pd.get_dummies(dataset.Sex, prefix='Sex')
merged = pd.concat([merged, dummies], axis='columns')
merged = merged.drop(['Sex'], axis='columns')

Drug_column = merged.pop('Drug')
merged['Drug'] = Drug_column

X = merged.iloc[:, :-1].values
y = merged.iloc[:, -1].values


modelBase=DecisionTreeClassifier(random_state=42)
modelGini = DecisionTreeClassifier(criterion='gini', random_state=42)
modelEntropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
modelLogLoss = DecisionTreeClassifier(criterion='log_loss', random_state=42)

crossVSBaseAccuracy=cross_val_score(modelBase, X, y, cv=10, scoring='accuracy')
crossVSGiniAccuracy=cross_val_score(modelBase, X, y, cv=10, scoring='accuracy')
crossVSEntropyAccuracy=cross_val_score(modelBase, X, y, cv=10, scoring='accuracy')
crossVSLogLossAccuracy=cross_val_score(modelBase, X, y, cv=10, scoring='accuracy')

for i in range(5):
    modelDepth = DecisionTreeClassifier(max_depth=i+1, random_state=42)
    crossVSDepthAccuracy = cross_val_score(modelDepth, X, y, cv=10, scoring='accuracy')
    if outputTableOfAccuracies:
        print(', '.join(map(str, crossVSDepthAccuracy)))
    print("Mean accuracy for Depth", i+1, "model,", crossVSDepthAccuracy.mean())

if outputTableOfAccuracies:
    print(', '.join(map(str, crossVSBaseAccuracy)))
print("Mean accuracy for Base model (no depth limit),", crossVSBaseAccuracy.mean())

if outputTableOfAccuracies:
    print(', '.join(map(str, crossVSGiniAccuracy)))
print("Mean accuracy for Gini model,", crossVSGiniAccuracy.mean())

if outputTableOfAccuracies:
    print(', '.join(map(str, crossVSEntropyAccuracy)))
print("Mean accuracy for Entropy model,", crossVSEntropyAccuracy.mean())

if outputTableOfAccuracies:
    print(', '.join(map(str, crossVSLogLossAccuracy)))
print("Mean accuracy for Log Loss model,", crossVSLogLossAccuracy.mean())