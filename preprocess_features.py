import numpy as np
import pandas as pd
from sklearn import preprocessing
import re
import math

raw_features = [
    # 'PassengerId', #0
    'Pclass', #1
    'Name', #2
    'Sex', #3
    'Age', #4
    'SibSp', #5
    'Parch', #6
    'Ticket', #7
    'Cabin', #8
    'Embarked' #9
]

raw_labels = [
    # 'PassengerId', #0
    'Survived', #1
]

def preprocess_instance(x):
    return [
        # x[0], #id
        x[0], #class
        re.search('\\s([A-Za-z]+)\\.', x[1]).group(1), #title
        x[2], #sex
        x[3], #age
        x[4] + x[5], #family_size
        re.sub('[0-9\\s\\/\\.]+', '', x[6]), #ticket_type
        '' if type(x[7]) != str and math.isnan(x[7]) else re.sub('[0-9]+', '', x[7]), #cabin_type
        # '' if type(x[8]) != str and math.isnan(x[8]) else re.search('[0-9]+', x[8]).group(0)
        x[8] #embarked
    ]

def preprocess_features(df):
    preprocessed1 = df.get(raw_features).values.tolist()
    preprocessed2 = map(preprocess_instance, preprocessed1)
    return preprocessed2

def preprocess_labels(df):
    preprocessed1 = df.get(raw_labels).values.tolist()
    return preprocessed1

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

preprocessed_train_df = preprocess_features(train_df)
preprocessed_test_df = preprocess_features(test_df)
preprocessed_labels = preprocess_labels(train_df)

all_features = preprocessed_train_df + preprocessed_test_df

all_features_df = pd.DataFrame(all_features)

def categorify_column(df, i):
    df[i] = df[i].astype('category')

def categorify():
    categorify_column(all_features_df, 0)
    categorify_column(all_features_df, 1)
    categorify_column(all_features_df, 2)
    categorify_column(all_features_df, 5)
    categorify_column(all_features_df, 6)
    categorify_column(all_features_df, 7)

    cat_columns = all_features_df.select_dtypes(['category']).columns
    all_features_df[cat_columns] = all_features_df[cat_columns].apply(lambda x: x.cat.codes)

categorify()

def one_hot_encode(v):
    _max = int(max(v)) + 1
    _min = int(min(v))
    _range = map(lambda x: [1 if n == x else 0 for n in range(_min, _max)], range(_min, _max))
    return map(lambda x: _range[int(x)], v)

def normalize_encode(v):
    avg = math.ceil(np.nansum(v) / len(v))
    v = map(lambda x: avg if math.isnan(x) else x, v)
    v = preprocessing.normalize([v]).tolist()[0]
    return v

def flatten(v):
    result = []
    for x in v:
        if type(x) == list:
            result += x
        else:
            result.append(x)
    return result

def encode():
    all_features_encoded = all_features_df.values.T.tolist()

    all_features_encoded[0] = one_hot_encode(all_features_encoded[0])
    all_features_encoded[1] = one_hot_encode(all_features_encoded[1])
    all_features_encoded[2] = one_hot_encode(all_features_encoded[2])
    all_features_encoded[5] = one_hot_encode(all_features_encoded[5])
    all_features_encoded[6] = one_hot_encode(all_features_encoded[6])
    all_features_encoded[7] = one_hot_encode(all_features_encoded[7])

    all_features_encoded[3] = normalize_encode(all_features_encoded[3])
    all_features_encoded[4] = normalize_encode(all_features_encoded[4])

    all_features_encoded = pd.DataFrame(all_features_encoded).values.T.tolist()
    all_features_encoded = map(flatten, all_features_encoded)

    return all_features_encoded

all_features_encoded = encode()

print(all_features_encoded[0])

all_features_encoded_train = all_features_encoded[0:len(preprocessed_labels)]
all_features_encoded_test = all_features_encoded[len(preprocessed_labels):]

np.savetxt('data/features_train.csv', all_features_encoded_train, fmt='%s', delimiter=',')
np.savetxt('data/features_test.csv', all_features_encoded_test, fmt='%s', delimiter=',')
np.savetxt('data/labels.csv', preprocessed_labels, fmt='%s', delimiter=',')
np.savetxt('data/test_ids.csv', test_df.get('PassengerId').values.tolist(), fmt='%d', delimiter=',')
