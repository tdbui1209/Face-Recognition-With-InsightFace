from insightface.app import FaceAnalysis


class Extractor:

    def __init__(self, model_name='buffalo_l', det_size=(640, 640)):
        self.app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=det_size)
    
    def get_embedding(self, img, det_threshold=0.5):
        faces = self.app.get(img)
        if len(faces) != 1:
            return None
        if faces[0].det_score >= det_threshold:
            return faces[0].embedding
