from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


def save_predictions(prediction, method):
    """ Saves predictions to a CSV file """
    predicted = np.column_stack((test.index.values, prediction))
    np.savetxt('pr_' + method + '.csv', predicted.astype(int), fmt='%d', delimiter=',',
               header='PassengerId,Survived', comments='')

train = pd.read_csv('cl_train.csv', index_col='PassengerId')
test = pd.read_csv('cl_test.csv', index_col='PassengerId')

# create training set X and y
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']

# combine,
tr_len = len(X_train)
X_e = X_train.append(test)

# create dummies, split, scale
X_d = pd.get_dummies(X_e, columns=['Sex', 'Pclass', 'Embarked'])
X_train_d = X_d[:tr_len]
X_test_d = X_d[tr_len:]
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_d)
X_test_s = scaler.transform(X_test_d)

# encode categorical variables
le = LabelEncoder()
X_e['Sex'] = le.fit_transform(X_e['Sex'])
X_e['Embarked'] = le.fit_transform(X_e['Embarked'])
X_train_e = X_e[:tr_len]
X_test_e = X_e[tr_len:]

# fit logistic polynomial regression and save predictions
poly = PolynomialFeatures(degree=2, include_bias=True)
lgr = LogisticRegression(C=0.03)
poly_lgr = Pipeline([('poly_features', poly), ('logistic', lgr)])
poly_lgr.fit(X_train_s, y_train)
save_predictions(poly_lgr.predict(X_test_s), 'logistic')

# fit SVM classifier and save predictions
svm = SVC(C=30, gamma='auto')
svm.fit(X_train_s, y_train)
save_predictions(svm.predict(X_test_s), 'svm')

# fit random forest classifier and save predictions
rf = RandomForestClassifier(n_estimators=400, max_depth=8)
rf.fit(X_train_e, y_train)
save_predictions(rf.predict(X_test_e), 'forest')
