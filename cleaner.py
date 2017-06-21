from sklearn.model_selection import train_test_split
import pandas as pd

# import and data cleaning

train = pd.read_csv('train.csv', header=0, index_col='PassengerId')
test = pd.read_csv('test.csv', header=0, index_col='PassengerId')

# function for data cleaning


def clean_data(df):
    # drop unused columns
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # normalizes fare by passenger class: null values receive zero
    df['Fare'] = df.groupby('Pclass')['Fare'].apply(lambda x: x.sub(x.mean()))
    df['Fare'] = df['Fare'].fillna(0)

    # fill null ages with median age (based on sex, passenger class)
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

    # split sex and passenger class into dummy variables
    df = pd.get_dummies(df, columns=['Sex', 'Pclass', 'Embarked'])

    # return cleaned data
    return df

train = clean_data(train)
test = clean_data(test)

# save cleaned data to CSV files
train.to_csv('train2.csv')
test.to_csv('test2.csv')

# create cross-validation set and CV training set
cvTrain, CV = train_test_split(train, test_size=0.3)
cvTrain.to_csv('cvTrain.csv')
CV.to_csv('cvSet.csv')
