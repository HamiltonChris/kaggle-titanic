import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation

def format_data():
    titanic = pandas.read_csv("../data/train.csv")

#print(titanic.describe())

# fill the empty values in the age column
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# encode sex values
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

#print(titanic["Embarked"].unique())
# fill empty embarked values and encode them with numbers
    titanic["Embarked"] = titanic["Embarked"].fillna("S")
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"] 
    target = ["Survived"]
    return titanic, predictors,target

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

    return predictions

def log_reg(dataset, features, target):
    alg = LogisticRegression()
    scores = cross_validation.cross_val_score(alg, dataset[features],dataset[target[0]], cv=3)
    return scores.mean()

def check_error(predictions, dataset, target):
    predictions = np.concatenate(predictions, axis=0)

    predictions[predictions > .5] = 1
    predictions[predictions <=.5] = 0
    accuracy = sum(predictions[predictions == dataset[target]]) / len(predictions)
    return accuracy
