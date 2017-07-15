# Titanic: Machine Learning From Disaster

My analysis of the Kaggle Titanic Dataset

#### v1.0:
* Drop `Name`, `Ticket` and `Cabin` columns.
* Transform `Fare` column to indicate difference from the median fare by passenger class.
* Imputes missing `Age` values with median based on sex and passenger class.
* Transform `SibSp` and `Parch` into `FamSize` feature by taking a sum.
* Scores:
    * Random Forest: 0.79904 with `n_estimators=300, max-depth=6`.
    * Logistic Regression: 0.77512 with `degree=3, C=0.005`.
    * SVM: 0.77033 with `gamma='auto', C=3`.

#### v1.1:
* Imputed using the median from both the train and test sets, instead of just the train set.
* Scores:
    * Random Forest: 0.78947 with `n_estimators=400, max-depth=7`.
    * Logistic Regression: 0.77033 with `degree=3, C=0.003`.
    * SVM: 0.78947 with `gamma='auto', C=1`.

#### v1.3:
* Added 1 to `FamSize` and log transformed `Fare`. Added `max_features` parameters for random forests.
* Scores:
    * Random Forest: 0.79426 with `n_estimators=400, max_features=4, max-depth=5`.
    * Logistic Regression: 0.77033 with `degree=2, C=0.03`.
    * SVM: 0.77990 with `gamma='auto', C=3`.

#### v2.0:
* Create a `Title` feature from `Name`. Then drop `Name`.
* Create a `TicketSize` feature: Size of each group sharing a `Ticket` number.
    * Divide `Fare` by `TicketSize` to get per-person fare.
    * Then drop `TicketSize` and `Ticket`.
* Impute `Age` values using median, grouped by `Title` and `Sex`.
    * Then create a `Child` feature and drop `Age`.
* Transform `SibSp` and `Parch` into `FamSize` by addition.
    * Then transform `FamSize` into `LargeFam` and `SmallFam` indicators.
* Drop the `Embarked` and `Cabin` features.