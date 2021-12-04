import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


ap = argparse.ArgumentParser()

ap.add_argument('--features', default='./model/X_test.npy',
                help='Path to matrix features')

ap.add_argument('--targets', default='./model/y_test.npy',
                help='Path to vector targets')

ap.add_argument('--model', default='./model/my_model.sav',
                help='Path to classifier model')

args = ap.parse_args()


X_test = np.load(args.features)
y_test = np.load(args.targets)

loaded_model = pickle.load(open(args.model, 'rb'))
labels = loaded_model.classes_

y_pred = loaded_model.predict(X_test)

print('Accuracy Score:', accuracy_score(y_test, y_pred))
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='viridis', xticklabels=labels, yticklabels=labels)
plt.show()
