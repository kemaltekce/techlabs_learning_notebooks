import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class dataset:

    def __init__(self, data):
        data.reset_index(drop=True, inplace=True)
        self.data = {'train': data}
        self.age_avg = data['Age'].mean()
        self.age_std = data['Age'].std()
        self.fare_avg = data['Fare'].median()

    def add_dataset(self, name, data):
        data.reset_index(drop=True, inplace=True)
        self.data[name] = data

    def clean(self, name='train'):
        df = self.data[name].copy()
        df = self.age_generator(df)
        # embarked
        df['Embarked'] = df['Embarked'].fillna('S')
        # fare
        df['Fare'] = df['Fare'].fillna(self.fare_avg)
        # drop uninteresting columns
        cols = ['Cabin', 'Name', 'PassengerId', 'Ticket']
        drop_cols = [x for x in df.columns if x in cols]
        df.drop(columns=drop_cols, inplace=True)
        self.data[name] = df

    def create_new_feature(self, name='train'):
        df = self.data[name].copy()
        # create family size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        # create is alone
        df['IsAlone'] = 0
        df.loc[df.FamilySize == 1, 'IsAlone'] = 1
        cols = ['Parch', 'SibSp']
        drop_cols = [x for x in df.columns if x in cols]
        df.drop(columns=drop_cols, inplace=True)
        self.data[name] = df

    def encode_labels(self, name='train'):
        df = self.data[name].copy()
        # TODO use for loop instead of duplicating code for sex and embarked
        if name == 'train':
            self.enc_sex = LabelEncoder()
            self.enc_embarked = LabelEncoder()
            self.enc_sex.fit(df['Sex'])
            self.enc_embarked.fit(df['Embarked'])
        df['Sex'] = self.enc_sex.transform(df['Sex'])
        df['Embarked'] = self.enc_embarked.transform(df['Embarked'])
        self.data[name] = df

    def create_dummies(self, name='train'):
        df = self.data[name].copy()
        # create dummy variables
        # TODO use for loop instead of duplicating code for pclass and embarked
        if name == 'train':
            self.oenc_pclass = OneHotEncoder()
            self.oenc_embarked = OneHotEncoder()
            self.oenc_pclass.fit(df.Pclass.values.reshape(-1, 1))
            self.oenc_embarked.fit(df.Embarked.values.reshape(-1, 1))
        pclass = self.oenc_pclass.transform(df.Pclass.values.reshape(-1, 1)).toarray()
        embarked = self.oenc_embarked.transform(df.Embarked.values.reshape(-1, 1)).toarray()
        pclass = pd.DataFrame(pclass)
        embarked = pd.DataFrame(embarked)
        for dummy in [('Pclass', pclass), ('Embarked', embarked)]:
            col_names = dummy[1].columns
            new_col_names = [dummy[0] + str(x) for x in col_names]
            name_dict = dict(zip(col_names, new_col_names))
            dummy[1].rename(columns=name_dict, inplace=True)
            df = df.join(dummy[1].iloc[:, :-1])
            df.drop(columns=[dummy[0]], inplace=True)
        self.data[name] = df

    def feature_scale(self, name='train'):
        df = self.data[name].copy()
        if name == 'train':
            self.scaler = StandardScaler()
            self.scaler.fit(df[['Age', 'Fare', 'FamilySize']])
        df[['Age', 'Fare', 'FamilySize']] = self.scaler.transform(
            df[['Age', 'Fare', 'FamilySize']])
        self.data[name] = df

    def age_generator(self, df):
        # we need the number of null values to know how much random ages to generate
        age_null_count = df['Age'].isnull().sum()
        # this will generate a list of random numbers between (mean - std) and (mean + std)
        age_null_random_list = np.random.randint(
            self.age_avg - self.age_std, self.age_avg + self.age_std, size=age_null_count)
        # select all nan ages and set it equal to the list of random ages
        df['Age'][np.isnan(df['Age'])] = age_null_random_list
        return df

    def preparation_pipeline(self, name):
        self.clean(name)
        self.create_new_feature(name)
        self.encode_labels(name)
        self.create_dummies(name)
        self.feature_scale(name)