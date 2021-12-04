import os
import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle
from my_utils.utils import output_name


ap = argparse.ArgumentParser()

ap.add_argument('--name', default='webface',
                help='Name of your model zoo')

ap.add_argument('--model', default='./model/my_model.sav',
                help='Path to classifier model')

ap.add_argument('--input-path', default='./testset/test.jpg',
                help='Path to input image')

ap.add_argument('--output-path', default='./output',
                help='Path to putput image')

ap.add_argument('--threshold', default=0.4,
                help='Threshold of recognize confident')

ap.add_argument('--font-size', default=0.7,
                help='Labels font size')

ap.add_argument('--bbox-thickness', default=1,
                help='Bounding box thickness')

ap.add_argument('--under-threshold', default='Unknown',
                help='Label if recognize confident under threshold')


args = ap.parse_args()

app = FaceAnalysis(name=args.name)

app.prepare(ctx_id=0, det_size=(640, 640))

loaded_model = pickle.load(open(args.model, 'rb'))
labels = loaded_model.classes_

img = cv2.imread(args.input_path)
faces = app.get(img)
dimg = img.copy()
for i in range(len(faces)):
    X = []
    face = faces[i]

    box = face.bbox.astype(np.int32)
    cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    y_prob = loaded_model.predict_proba(face.embedding.reshape(1, -1))[0]
    if max(y_prob) > args.threshold:
        label = labels[np.argmax(y_prob)]
    else:
        label = args.under_threshold

    cv2.putText(
        dimg, label,
        (box[0] - 1, box[1] - 4),
        cv2.FONT_HERSHEY_COMPLEX, args.font_size, (0, 255, 0), args.bbox_thickness)

cv2.imshow('Face Recognize', dimg)
cv2.waitKey(0)

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

name = output_name(args.output_path, args.input_path, 'jpg')
cv2.imwrite(os.path.join(args.output_path, name), dimg)

cv2.destroyAllWindows()
