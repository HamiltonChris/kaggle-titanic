import pandas
import sklearn

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

print(titanic.describe())
