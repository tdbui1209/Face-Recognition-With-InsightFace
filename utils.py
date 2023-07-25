from insightface.app import FaceAnalysis
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class Extractor:
    '''
    Extract embeddings from images
    '''
    def __init__(self, model_name='buffalo_l', det_size=(640, 640)):
        self.app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=det_size)
    
    def get_embedding(self, img, det_threshold=0.5):
        faces = self.app.get(img)
        if len(faces) != 1:
            return None
        if faces[0].det_score >= det_threshold:
            return faces[0].embedding


class ThresholdFinder:

    @staticmethod
    def find_global_threshold(X_valid, y_valid, model, step=0.01):
        '''
        Find the best threshold for the whole validation set
        '''
        thresholds = []
        accuracy = []
        for threshold in tqdm(np.arange(step, 1 + step, step)):
            y_pred = model.predict_proba(X_valid) >= threshold
            y_pred = model.classes_[y_pred.argmax(axis=1)]
            acc = accuracy_score(y_valid, y_pred)
            thresholds.append(threshold)
            accuracy.append(acc)
        return thresholds[np.argmax(accuracy)]

    @staticmethod
    def find_local_thresholds(valid_df, y_valid, y_pred, model, step=0.01):
        '''
        Find the best threshold for each id in the validation set
        '''
        local_thresholds = {}
        for _id in tqdm(valid_df['id'].unique()):
            indices = [i for i, x in enumerate(y_valid) if x == _id]
            y_pred_of_id = y_pred[indices]
            max_threshold, max_acc = 0, 0
            for threshold in np.arange(step, 1 + step, step):
                y_pred_of_id = y_pred_of_id >= threshold
                y_pred_of_id = y_pred_of_id.argmax(axis=1)
                y_pred_of_id = [model.classes_[i] for i in y_pred_of_id]
                acc = accuracy_score(y_valid[indices], y_pred_of_id)
                if acc > max_acc:
                    max_threshold = threshold
                    max_acc = acc
            local_thresholds[_id] = [max_threshold, max_acc]
        return local_thresholds
