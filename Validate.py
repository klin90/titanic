from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def log_grid_search(params):
    """ Runs a grid search for polynomial logistic regression with given parameters. """
    grid = GridSearchCV(p_lgr, param_grid=params, cv=10).fit(X_s, y)
    print(grid.cv_results_['mean_test_score'].reshape(len(params['logistic__C']),
                                                      len(params['poly_features__degree'])))
    print('Best Parameters: %s' % grid.best_params_)


def rf_grid_search(params):
    """ Runs a grid search for polynomial logistic regression with given parameters. """
    grid = GridSearchCV(rf, param_grid=params, cv=10).fit(X_s, y)
    print(grid.cv_results_['mean_test_score'].reshape(len(params['max_depth']),
                                                      len(params['max_features'])))
    print('Best Parameters: %s' % grid.best_params_)


def grid_search(clf, params):
    """ Runs a grid search for given classifier and parameters. """
    grid = GridSearchCV(clf, param_grid=params, cv=10).fit(X_s, y)
    print(grid.cv_results_['mean_test_score'])
    print('Best Parameters: %s' % grid.best_params_)

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

# logistic regression
poly = PolynomialFeatures(include_bias=True)
lgr = LogisticRegression()
p_lgr = Pipeline([('poly_features', poly), ('logistic', lgr)])
p_lgr_params = {'poly_features__degree': [2, 3, 4],
                'logistic__C': [0.003, 0.01, 0.03, 0.1, 1]}
print('... Logistic Regression Search...')
log_grid_search(p_lgr_params)

# SVM grid search
svm = SVC(gamma='auto')
svm_params = {'C': [3, 10, 30, 100, 300]}
grid_search(svm, svm_params)

# random forest grid search
rf = RandomForestClassifier(n_estimators=400)
rf_params = {'max_depth': [5, 6, 7, 8, 9],
             'max_features': [4, 5, 6]}
rf_grid_search(rf_params)
