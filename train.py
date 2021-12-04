import os
import argparse
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


ap = argparse.ArgumentParser()

ap.add_argument('--features', default='./model/X.npy',
                help='Path to matrix features')

ap.add_argument('--targets', default='./model/y.npy',
                help='Path to vector targets')

ap.add_argument('--output', default='./model',
                help='Path to classifier model')

ap.add_argument('--name', default='my_model',
                help='Name of classifier model')

args = ap.parse_args()

X = np.load(args.features)
y = np.load(args.targets)

# model = RandomForestClassifier(random_state=42)
# model.fit(X, y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

pickle.dump(model, open(os.path.join(args.output, args.name + '.sav'), 'wb'))
