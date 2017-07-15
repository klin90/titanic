import pandas as pd

train = pd.read_csv('train.csv', index_col='PassengerId')
test = pd.read_csv('test.csv', index_col='PassengerId')

# create X train
X_train = train.drop('Survived', axis=1)

# combine X train and test for pre-processing
tr_len = len(X_train)
df = X_train.append(test)

# create 'Title' feature
df['Title'] = df['Name'].str.extract('\,\s(.*?)[.]', expand=False)
df['Title'].replace('Mme', 'Mrs', inplace=True)
df['Title'].replace('Mlle', 'Miss', inplace=True)
df['Title'].replace('Ms', 'Miss', inplace=True)
df['Title'].replace('Lady', 'fNoble', inplace=True)
df['Title'].replace('the Countess', 'fNoble', inplace=True)
df['Title'].replace('Dona', 'fNoble', inplace=True)
df['Title'].replace('Don', 'mNoble', inplace=True)
df['Title'].replace('Sir', 'mNoble', inplace=True)
df['Title'].replace('Jonkheer', 'mNoble', inplace=True)
df['Title'].replace('Col', 'mil', inplace=True)
df['Title'].replace('Capt', 'mil', inplace=True)
df['Title'].replace('Major', 'mil', inplace=True)
df.drop('Name', axis=1, inplace=True)

# create FamSize feature
df['FamSize'] = df['SibSp'] + df['Parch']
df['SmallFam'] = (df['FamSize'] < 4) & (df['FamSize'] > 0)
df['LargeFam'] = df['FamSize'] >= 4
df.drop(['SibSp', 'Parch', 'FamSize'], axis=1, inplace=True)

# create 'TicketSize' and 'Fare' features
df['TicketSize'] = df['Ticket'].value_counts()[df['Ticket']].values
df['Fare'] = df['Fare'].div(df['TicketSize'])
df['Fare'] = df.groupby('Pclass')['Fare'].apply(lambda x: x.fillna(x.median()))
df.drop('Ticket', axis=1, inplace=True)

# impute missing 'Age' values with median by 'Title', 'Sex'
df['Age'] = df.groupby(['Sex', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))

# create 'Child' feature
df['Child'] = df['Age'] <= 12
df.drop('Age', axis=1, inplace=True)

# drop other unused features
df.drop(['Cabin', 'Embarked'], axis=1, inplace=True)

# split X train and test
X_train = df[:tr_len]
test = df[tr_len:]

# recreate train DataFrame
train = X_train.join(train[['Survived']])

# save transformed data
train.to_csv('cl_train.csv')
test.to_csv('cl_test.csv')
