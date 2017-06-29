from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv('cl_train.csv', index_col='PassengerId')

# create training set X and y
X_e = df.drop('Survived', axis=1)
y = df['Survived']

# create dummies, split, scale
X_d = pd.get_dummies(X_e, columns=['Sex', 'Pclass', 'Embarked'])
scaler = StandardScaler()
X_s = scaler.fit_transform(X_d)

# encode categorical variables
le = LabelEncoder()
X_e['Sex'] = le.fit_transform(X_e['Sex'])
X_e['Embarked'] = le.fit_transform(X_e['Embarked'])


# logistic grid search
def logistic_grid():
    """ Polynomial Logistic Regression Grid Search """
    poly = PolynomialFeatures(include_bias=True)
    lgr = LogisticRegression()
    poly_lgr = Pipeline([('poly_features', poly), ('logistic', lgr)])
    params = {'poly_features__degree': [2, 3, 4],
              'logistic__C': [0.003, 0.01, 0.03, 0.1, 1]}
    grid = GridSearchCV(poly_lgr, param_grid=params, cv=10, n_jobs=2).fit(X_s, y)
    print(grid.cv_results_['mean_test_score'].reshape(5, 3))
    print('Best Parameters: %s' % grid.best_params_)


# SVM grid search
def svm_grid():
    """ SVC Grid Search """
    svm = SVC(gamma='auto')
    params = {'C': [3, 10, 30, 100, 300]}
    grid = GridSearchCV(svm, param_grid=params, cv=10, n_jobs=2).fit(X_s, y)
    print(grid.cv_results_['mean_test_score'])
    print('Best Parameters: %s' % grid.best_params_)


# random forest grid search
def rf_grid():
    """ RF Classifier Grid Search """
    rf = RandomForestClassifier(n_estimators=400)
    params = {'max_depth': [5, 6, 7, 8, 9, 10, 11]}
    grid = GridSearchCV(rf, param_grid=params, cv=10, n_jobs=2).fit(X_s, y)
    print(grid.cv_results_['mean_test_score'])
    print('Best Parameters: %s' % grid.best_params_)
