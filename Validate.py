from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
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
X = df.drop('Survived', axis=1)
y = df['Survived']

# create dummies, split, scale
X_d = pd.get_dummies(X, columns=['Sex', 'Pclass', 'Embarked'])
X_train_d, X_test_d, y_train, y_test = train_test_split(X_d, y, random_state=14)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_d)
X_test_s = scaler.transform(X_test_d)

# encode categorical variables
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])
X['Embarked'] = le.fit_transform(X['Embarked'])
X_train_e, X_test_e, y_train, y_test = train_test_split(X, y, random_state=14)


# logistic grid search
def logistic_grid():
    """ Polynomial Logistic Regression Grid Search """
    poly = PolynomialFeatures(include_bias=True)
    lgr = LogisticRegression()
    poly_lgr = Pipeline([('poly_features', poly), ('logistic', lgr)])
    params = {'poly_features__degree': [2, 3, 4],
              'logistic__C': [0.001, 0.003, 0.01, 0.03, 0.1, 1]}
    grid = GridSearchCV(poly_lgr, param_grid=params, cv=10, n_jobs=2).fit(X_train_s, y_train)
    print(grid.cv_results_['mean_test_score'].reshape(6, 3))
    print('Best Parameters: %s' % grid.best_params_)
    print('Final Score: %s' % grid.score(X_test_s, y_test))


# SVM grid search
def svm_grid():
    """ SVC Grid Search """
    svm = SVC(gamma='auto')
    params = {'C': [0.1, 0.3, 1, 3, 10, 30]}
    grid = GridSearchCV(svm, param_grid=params, cv=5, n_jobs=2).fit(X_train_s, y_train)
    print(grid.cv_results_['mean_test_score'])
    print('Best Parameters: %s' % grid.best_params_)
    print('Final Score: %s' % grid.score(X_test_s, y_test))


# random forest grid search
def rf_grid():
    """ RF Classifier Grid Search """
    rf = RandomForestClassifier(n_estimators=400)
    params = {'max_depth': [5, 6, 7, 8, 9, 10, 11]}
    grid = GridSearchCV(rf, param_grid=params, cv=5, n_jobs=2).fit(X_train_s, y_train)
    print(grid.cv_results_['mean_test_score'])
    print('Best Parameters: %s' % grid.best_params_)
    print('Final Score: %s' % grid.score(X_test_s, y_test))
