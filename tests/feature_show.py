import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import pickle


def show(rf):
    scores = rf.feature_importances_
    names = []
    res_scores = []
    for i in range(len(scores)):
        if scores[i] > 0.0035:
            names.append(i)
            res_scores.append(scores[i])
            print(i, ": ", scores[i])
    print("feature_length", len(names))
    plt.barh(names, res_scores)
    plt.xlabel("Feature Importance")
    plt.show()


if __name__ == '__main__':
    with open('models/best.pickaim', 'rb') as f:
        model = pickle.load(f)
        show(model)
