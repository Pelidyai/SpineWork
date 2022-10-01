import pickle

import cv2


class VertForestModel:
    def __init__(self, filename):
        self.model = pickle.load(open(filename, 'rb'))

    @staticmethod
    def get_x(img) -> []:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (112, 126), interpolation=cv2.INTER_LINEAR)
        result = []
        for i in range(0, len(img), 5):
            for j in range(0, len(img[i]), 5):
                result.append(img[i][j])
        return result

    def predict(self, img):
        return self.model.predict([self.get_x(img)])[0]


class VertNNModel:
    def __init__(self, mask_model_filename, filename):
        self.indexes = []
        self.model = pickle.load(open(filename, 'rb'))
        self.get_feature_indexes(pickle.load(open(mask_model_filename, 'rb')), 0.0075)

    def get_feature_indexes(self, rf, rate):
        scores = rf.feature_importances_
        out = {}
        while len(out) < self.model.n_features_in_:
            for i in range(len(scores)):
                if len(out) == self.model.n_features_in_:
                    break
                if scores[i] > rate:
                    out[i] = scores[i]
            rate -= 0.001
        self.indexes = list(out.keys())

    def get_featuring_x(self, img):
        all_x = self.get_x(img)
        return [all_x[i] for i in self.indexes]

    @staticmethod
    def get_x(img) -> []:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (112, 126), interpolation=cv2.INTER_LINEAR)
        result = []
        for i in range(0, len(img), 5):
            for j in range(0, len(img[i]), 5):
                result.append(img[i][j])
        return result

    def predict(self, img):
        return self.model.predict([self.get_featuring_x(img)])[0]
