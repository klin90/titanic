from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

################################
# logistic polynomial regression

##################
# cross-validation

# train = pd.read_csv('cvTrain.csv', header=0, index_col='PassengerId')
# cvset = pd.read_csv('cvSet.csv', header=0, index_col='PassengerId')

# # select all features + create numpy arrays
# train_X = train.drop('Survived', axis=1).values
# train_Y = train['Survived'].values
# cvset_X = cvset.drop('Survived', axis=1).values
# cvset_Y = cvset['Survived'].values

# # normalize age (column 0)
# scaler = StandardScaler()
# scaler.fit(train_X[:, 0].reshape(-1, 1))
# train_X[:, 0] = scaler.transform([train_X[:, 0]])[0]
# cvset_X[:, 0] = scaler.transform([cvset_X[:, 0]])[0]

# # L2 logistic polynomial regression with C = 1
# polynomial_features = PolynomialFeatures(degree=2, include_bias=True)
# logistic_regression = LogisticRegression(C=1.0)
# pipeline = Pipeline([('polynomial_features', polynomial_features), ('logistic_regression', logistic_regression)])

# # fit and predict
# pipeline.fit(train_X, train_Y)
# print(pipeline.score(cvset_X, cvset_Y))

##########
# test set

train = pd.read_csv('train2.csv', header=0, index_col='PassengerId')
test = pd.read_csv('test2.csv', header=0, index_col='PassengerId')

# select all features + create numpy arrays
train_X = train.drop('Survived', axis=1).values
train_Y = train['Survived'].values
test_X = test.values

# normalize age (column 0)
scaler = StandardScaler()
scaler.fit(train_X[:, 0].reshape(-1, 1))
train_X[:, 0] = scaler.transform([train_X[:, 0]])[0]
test_X[:, 0] = scaler.transform([test_X[:, 0]])[0]

# L2 logistic polynomial regression with C = 1
polynomial_features = PolynomialFeatures(degree=2, include_bias=True)
logistic_regression = LogisticRegression(C=1.0)
pipeline = Pipeline([('polynomial_features', polynomial_features), ('logistic_regression', logistic_regression)])

# fit and predict
pipeline.fit(train_X, train_Y)
prediction = pipeline.predict(test_X)

# save survival predictions to a CSV file
predicted = np.column_stack((test.index.values, prediction))
np.savetxt("logistic.csv", predicted.astype(int), fmt='%d', delimiter=",", header="PassengerId,Survived", comments='')
