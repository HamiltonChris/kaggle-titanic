import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation

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

    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"] 
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
