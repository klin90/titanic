# Titanic: Machine Learning From Disaster

My analysis of the Kaggle Titanic Dataset

#### R1:
* Drop `Name`, `Ticket` and `Cabin` columns.
* Transform `Fare` column to indicate difference from the median fare by passenger class.
* Imputes missing `Age` values with median based on sex and passenger class.
* Transform `SibSp` and `Parch` into `FamSize` feature by taking a sum.
* Scores:
    * Random Forest: 0.79904 with `n_estimators=300` and `max-_depth=6`.
    * Logistic Regression: 0.77512 with `degree=3` and `C=0.005`.
    * SVM: 0.77033 with `gamma='auto'` and `C=5`.