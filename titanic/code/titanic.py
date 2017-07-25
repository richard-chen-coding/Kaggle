import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
import re

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""



warnings.filterwarnings('ignore')

data_train = pd.read_csv("..\\data\\train.csv")
data_test = pd.read_csv("..\\data\\test.csv")

data_train['source'] = 'train'
data_test['source']= 'test'
fulldata = pd.concat([data_train, data_test], ignore_index=True)


fulldata["Embarked"] = fulldata["Embarked"].fillna("C")
fulldata["Fare"] = fulldata["Fare"].fillna(8.05)
#normalize the fare
scaler = StandardScaler()
fulldata['Fare'] = pd.Series(scaler.fit_transform(fulldata.Fare.reshape(-1, 1)).reshape(-1), index=fulldata.index)



fulldata["FamilySize"] = fulldata["SibSp"] + fulldata["Parch"] + 1
fulldata["Family"] = 0
fulldata.set_value(fulldata.FamilySize == 1, 'Family', '0')
fulldata.set_value((fulldata.FamilySize > 1) & (fulldata.FamilySize <5), 'Family', '1')
fulldata.set_value((fulldata.FamilySize > 4) & (fulldata.FamilySize <8), 'Family', '2')
fulldata.set_value(fulldata.FamilySize > 7, 'Family', '3')

fulldata.Sex = np.where(fulldata.Sex == 'female', 0, 1)
fulldata = pd.get_dummies(fulldata, columns=['Embarked', 'Pclass', 'Family','Sex'])



titles = fulldata["Name"].apply(get_title)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                 "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2,
                 "Dona": 10}
for k, v in title_mapping.items():
    titles[titles == k] = v

fulldata["Title"] = titles


# better way to do this?
# base on title?
fulldata["Age"] = fulldata["Age"].fillna(fulldata["Age"].median())


test_input = fulldata[fulldata['source'] == 'test']
test_input.drop(labels=['PassengerId', 'Name', 'Cabin','Ticket',
                         'SibSp','Parch','FamilySize','source','Survived'], axis=1, inplace=True)

train_input = fulldata[fulldata['source'] == 'train']
train_input.drop(labels=['PassengerId', 'Name', 'Cabin','Ticket',
                         'SibSp','Parch','FamilySize','source','Survived'], axis=1, inplace=True)
train_output = fulldata[fulldata['source'] == 'train']["Survived"]


clf_GradingBoost = GradientBoostingClassifier(random_state=1, n_estimators=25)
clf_LogRegression = LogisticRegression(random_state=1)
clf_RandomForest = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

clfs = [clf_GradingBoost, clf_LogRegression, clf_RandomForest]

full_predictions = []

for clf in clfs:
    #scores = cross_validation.cross_val_score(clf, train_input, train_output, cv=3)
    #print scores
    clf.fit(train_input, train_output)

    predictions = clf.predict_proba(test_input)[:,1]
    full_predictions.append(predictions)

predictions = (full_predictions[0]  + full_predictions[1] + full_predictions[2]) / 3
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
df = pd.DataFrame(index = data_test['PassengerId'], data = predictions, columns=['Survived'])
#data_test['Survived'] = predictions
#df = data_test[['PassengerId', 'Survived']]
#df.index=df['PassengerId']
#df = pd.DataFrame(data=[test_input.index, predictions], index = test_input.index, columns=['PassengerId', 'Survived'])
df.to_csv("..\\data\\test_result.csv")