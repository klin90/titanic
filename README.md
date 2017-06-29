# Titanic: Machine Learning From Disaster

My analysis of the Kaggle Titanic Dataset

#### v1.0:
* Drop `Name`, `Ticket` and `Cabin` columns.
* Transform `Fare` column to indicate difference from the median fare by passenger class.
* Imputes missing `Age` values with median based on sex and passenger class.
* Transform `SibSp` and `Parch` into `FamSize` feature by taking a sum.
* Scores:
    * Random Forest: 0.79904 with `n_estimators=300` and `max-depth=6`.
    * Logistic Regression: 0.77512 with `degree=3` and `C=0.005`.
    * SVM: 0.77033 with `gamma='auto'` and `C=3`.

#### v1.1:
* Median values are calculated using both train and test sets, instead of just train set.
* Scores:
    * Random Forest: 0.78947 with `n_estimators=400` and `max-depth=7`.
    * Logistic Regression: 0.77033 with `degree=3` and `C=0.003`.
    * SVM: 0.78947 with `gamma='auto'` and `C=1`.

#### v1.2:
* Fare transformation now takes median by passenger class and embark location.
* No improvement, reverting back to v1.1.
* Scores:
    * Random Forest: 0.79426 with `n_estimators=400` and `max-depth=6`.
    * Logistic Regression: 0.76555 with `degree=2` and `C=0.03`.
    * SVM: 0.77033 with `gamma='auto'` and `C=3`.