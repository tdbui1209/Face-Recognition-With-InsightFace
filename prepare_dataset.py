import os
import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis


ap = argparse.ArgumentParser()

ap.add_argument('--name', default='webface',
                help='Name of your model zoo')

ap.add_argument('--data-path', default='./dataset',
                help='Path to training dataset')

ap.add_argument('--output', default='./model',
                help='Path to output features, targets')


args = ap.parse_args()

app = FaceAnalysis(name=args.name)
app.prepare(ctx_id=0, det_size=(640, 40))

data_path = args.data_path
file_names = [i for i in os.listdir(data_path)]

X = []
y = []

ext_names = ['.jpg', '.png', '.jpeg']

for file_name in file_names:
    images = ['.'.join(i.split('.')[:-1]) for i in os.listdir(os.path.join(data_path, file_name))]
    for image in images:

        for ext_name in ext_names:
            image_file = os.path.join(data_path, file_name, image + ext_name)
            if os.path.exists(image_file):
                img = cv2.imread(image_file)
                faces = app.get(img)
                if len(faces) != 1:
                    raise ValueError(f"{os.path.join(data_path, file_name, image)} doesn't have exactly one face")

                face = faces[0]
                X.append(face.embedding)
                y.append(file_name)
                break

X = np.array(X)
y = np.array(y)

np.save(os.path.join(args.output, 'X.npy'), X)
np.save(os.path.join(args.output, 'y.npy'), y)

cv2.destroyAllWindows()
