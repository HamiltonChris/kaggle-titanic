import pandas
import numpy as np
import re
import operator
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn import cross_validation


family_id_mapping = {}

# extracts a title if any from name
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def get_family_id(row):
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

def format_data():
    titanic = pandas.read_csv("../data/train.csv")
    titanic_test = pandas.read_csv("../data/test.csv")

# fill the empty values in the age column
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

# encode sex values
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
    titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 0

#print(titanic["Embarked"].unique())
# fill empty embarked values and encode them with numbers
    titanic["Embarked"] = titanic["Embarked"].fillna("S")
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
    titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
    titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
    titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

# fill missing fare values in the test data
    titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

# create new feature "familysize"
    titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
    titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

    
# create new feature "NameLength"
    titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
    titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x)) 

# retrieve titles from "Name" and create feature "Title"
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

    titles = titanic["Name"].apply(get_title)
    titles_test = titanic_test["Name"].apply(get_title)

    for k,v in title_mapping.items():
        titles[titles == k] = v
        titles_test[titles_test == k] = v

# family groups
    family_ids = titanic.apply(get_family_id, axis=1)
    family_ids[titanic["FamilySize"] < 3] = -1
    titanic["FamilyId"] = family_ids 
    
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","FamilySize","NameLength","Title"] 
    target = ["Survived"]
    return titanic, predictors,target, titanic_test

def linear_reg(dataset, features, target):
    alg = LinearRegression()

    kf = KFold(dataset.shape[0], n_folds = 3, random_state=1)
    predictions = []
    for train, test in kf:
        train_predictors = (dataset[features].iloc[train,:])
        train_target = dataset[target].iloc[train,:]
        alg.fit(train_predictors,train_target)
        test_predictions = alg.predict(dataset[features].iloc[test,:])
        predictions.append(test_predictions) 

    return predictions, alg


def log_reg(dataset, features, target):
    alg = LogisticRegression()
    scores = cross_validation.cross_val_score(alg, dataset[features],dataset[target[0]], cv=3)
    return scores.mean(), alg

def random_forest(dataset, features, target):
   alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2) 
   scores = cross_validation.cross_val_score(alg, dataset[features], dataset[target[0]], cv=3)
   return scores.mean(), alg

def check_error(predictions, dataset, target):
    predictions = np.concatenate(predictions, axis=0)

    predictions[predictions > .5] = 1
    predictions[predictions <=.5] = 0
    accuracy = sum(predictions[predictions == dataset[target]]) / len(predictions)
    return accuracy

def test_prediction(training_dataset, test_dataset, alg, predictors, target):
    alg.fit(training_dataset[predictors], training_dataset[target[0]])
    predictions = alg.predict(test_dataset[predictors]) 
    
    return predictions

def submit(test_set, predictions):
    submission = pandas.DataFrame({
        "PassengerId": test_set["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv("kaggle.csv", index=False)
