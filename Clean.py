import pandas as pd

train = pd.read_csv('train.csv', index_col='PassengerId')
test = pd.read_csv('test.csv', index_col='PassengerId')

# create X train
X_train = train.drop('Survived', axis=1)

# combine X train and test for pre-processing
tr_len = len(X_train)
df = X_train.append(test)

# drop unused columns
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# transform 'Fare' into difference from median by 'Pclass'
# impute missing 'Fare' values with 0
df['Fare'] = df.groupby('Pclass')['Fare'].apply(lambda x: x.sub(x.median()))
df['Fare'].fillna(0, inplace=True)

# impute missing 'Age' values with median by 'Sex', 'Pclass'
df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

# impute missing 'Embarked' values with 'S' (most frequent)
df['Embarked'].fillna(value='S', inplace=True)

# create FamSize feature
df['FamSize'] = df['SibSp'] + df['Parch']
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# split X train and test
X_train = df[:tr_len]
test = df[tr_len:]

# recreate train DataFrame
train = X_train.join(train[['Survived']])

# save transformed data
train.to_csv('cl_train.csv')
test.to_csv('cl_test.csv')
