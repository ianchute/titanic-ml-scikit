import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.metrics import accuracy_score
import math
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn import svm, neighbors

seed = 42

np.random.seed(seed)

def flatten(v):
    result = []
    for x in v:
        if type(x) == list:
            result += x
        else:
            result.append(x)
    return result

X = pd.read_csv('data/features_train.csv', sep=',',header=None).values.tolist()
y = flatten( pd.read_csv('data/labels.csv', sep=',',header=None).values.tolist() )
unknowns = pd.read_csv('data/features_test.csv', sep=',',header=None).values.tolist()

assert len(X) == len(y)

perm = np.random.permutation(len(X))

X = np.array(X)[perm]
y = np.array(y)[perm]

nslice = 600

X_train = X[0:nslice]
X_test = X[nslice:]
y_train = y[0:nslice]
y_test = y[nslice:]


rbm1 = BernoulliRBM(random_state=seed, verbose=True, n_iter=200, n_components=128)
rbm1.fit(X_test.tolist() + unknowns)

final = MLPClassifier(
    solver='sgd',
    hidden_layer_sizes=(64, 2),
    random_state=seed,
    max_iter=200,
    learning_rate="adaptive"
)
clf = Pipeline(steps=[('rbm1', rbm1), ('final', final)])
model = clf.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

test_ids = flatten( pd.read_csv('data/test_ids.csv', sep=',',header=None).values.tolist() )
unknown_pred = model.predict(unknowns).tolist()
predictions = pd.DataFrame([test_ids, unknown_pred]).values.T.tolist()
np.savetxt('data/predictions.csv', predictions, fmt='%d', delimiter=',', header='PassengerId,Survived')
