import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle


ap = argparse.ArgumentParser()

ap.add_argument('--name', default='webface',
                help='Name of your model zoo')

ap.add_argument('--model', default='my_model.sav',
                help='Path to classifier model')

ap.add_argument('--id-cam', default=0,
                help='Cam ID')

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

capture = cv2.VideoCapture(int(args.id_cam))
while True:
    _, frame = capture.read()
    faces = app.get(frame)
    dimg = frame.copy()
    for i in range(len(faces)):
        face = faces[i]

        box = face.bbox.astype(np.int32)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        y_prob = loaded_model.predict_proba(face.embedding.reshape(1, -1))[0]
        if max(y_prob) > float(args.threshold):
            label = labels[np.argmax(y_prob)]
        else:
            label = args.under_threshold

        cv2.putText(
            dimg, label,
            (box[0] - 1, box[1] - 4),
            cv2.FONT_HERSHEY_COMPLEX, float(args.font_size), (0, 255, 0), int(args.bbox_thickness))

    cv2.imshow('Face Recognize', dimg)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
