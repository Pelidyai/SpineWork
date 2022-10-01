import os
import cv2
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from points_detector import PointsDetector

from verts_models.mask_models import MaskForestModel, MaskNNModel
from verts_models.vert_models import VertForestModel, VertNNModel


def get_x(img) -> []:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (112, 126), interpolation=cv2.INTER_LINEAR)
    result = []
    for i in range(0, len(img), 5):
        for j in range(0, len(img[i]), 5):
            result.append(img[i][j])
    return result


def get_y(filename: str) -> np.array:
    result = []
    with open(filename, 'r') as file:
        val = file.readline()
        while val:
            for digit in val.split(" "):
                digit = digit.replace('\n', '')
                result.append(int(digit))
            val = file.readline()
    return np.array(result)


def get_xy(img_dir, points_dir) -> tuple[np.array, np.array]:
    x = []
    y = []
    for filename in os.listdir(img_dir):
        filename = filename.split('.')[0]
        img = cv2.imread(os.path.join(img_dir, filename + '.jpg'))
        x.append(get_x(img))
        y.append(get_y(os.path.join(points_dir, filename + '.txt')))
    return np.array(x), np.array(y)


def get_feature_indexes(rf, rate):
    scores = rf.feature_importances_
    names = []
    for i in range(len(scores)):
        if scores[i] > rate:
            names.append(i)
    print(len(names))
    return names


def get_featuring_x(img, indexes):
    all_x = get_x(img)
    return [all_x[i] for i in indexes]


def get_featuring_xy(img_dir, points_dir, base_model_file, feature_threshold):
    x = []
    y = []
    with open(base_model_file, 'rb') as f:
        model = pickle.load(f)
        indexes = get_feature_indexes(model, feature_threshold)
    for filename in os.listdir(img_dir):
        filename = filename.split('.')[0]
        img = cv2.imread(os.path.join(img_dir, filename + '.jpg'))
        x.append(get_featuring_x(img, indexes))
        y.append(get_y(os.path.join(points_dir, filename + '.txt')))
    return np.array(x), np.array(y)


def get_merged_x(mask_img, vert_img):
    result = []
    mask_forest_model = MaskForestModel('mask_forest_models/best.pickaim')
    mask_nn_model = MaskNNModel('mask_forest_models/best.pickaim', 'mask_nn_models/best.pickaim')
    vert_forest_model = VertForestModel('vert_forest_models/best.pickaim')
    vert_nn_model = VertNNModel('vert_forest_models/best.pickaim', 'vert_nn_models/best.pickaim')
    result.extend(mask_forest_model.predict(mask_img))
    result.extend(mask_nn_model.predict(mask_img))
    result.extend(vert_forest_model.predict(vert_img))
    result.extend(vert_nn_model.predict(vert_img))
    return result


def get_merged_xy(mask_img_dir, vert_img_dir, points_dir):
    x = []
    y = []
    for filename in os.listdir(mask_img_dir):
        filename = filename.split('.')[0]
        mask_img = cv2.imread(os.path.join(mask_img_dir, filename + '.jpg'))
        vert_img = cv2.imread(os.path.join(vert_img_dir, filename + '.jpg'))
        x.append(get_merged_x(mask_img, vert_img))
        y.append(get_y(os.path.join(points_dir, filename + '.txt')))
    return np.array(x), np.array(y)


def train(x, y, save_dir, model, n=100):
    min_error = get_min_model_error(save_dir)
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        model.fit(x_train, y_train)
        print("iteration#", i + 1, "___________________________")
        valid_error = mean_absolute_error(y_test, model.predict(x_test))
        print("valid MAE:", valid_error)
        error = mean_absolute_error(y_train, model.predict(x_train))
        print("train MAE:", error)
        print("best MAE:", min_error)
        if valid_error < min_error:
            filename = os.path.join(save_dir, 'best' + str(round(valid_error, 2)) + '.pickaim')
            pickle.dump(model, open(filename, 'wb'))
            filename = os.path.join(save_dir, 'best.pickaim')
            pickle.dump(model, open(filename, 'wb'))
            min_error = valid_error


def train_mask_forest(n_iteration=100):
    print("______________mask_forest_train______________")
    x, y = get_xy('../vert_masks', '../points')
    train(x, y, 'mask_forest_models',
          RandomForestRegressor(n_estimators=150, criterion="squared_error"), n=n_iteration)


def train_mask_nn(n_iteration=100):
    print("______________mask_nn_train______________")
    x, y = get_featuring_xy('../vert_masks', '../points', 'mask_forest_models/best.pickaim', 0.02)
    train(x, y, 'mask_nn_models', MLPRegressor(), n=n_iteration)


def train_vert_forest(n_iteration=100):
    print("______________vert_forest_train______________")
    x, y = get_xy('../verts', '../points')
    train(x, y, 'vert_forest_models',
          RandomForestRegressor(n_estimators=150, criterion="squared_error"), n=n_iteration)


def train_vert_nn(n_iteration=100):
    print("______________vert_nn_train______________")
    x, y = get_featuring_xy('../verts', '../points', 'vert_forest_models/best.pickaim', 0.0075)
    train(x, y, 'vert_nn_models', MLPRegressor(), n=n_iteration)


def train_merge(n_iteration=100):
    print("_______________merge_train___________________")
    x, y = get_merged_xy('../vert_masks', '../verts', '../points')
    train(x, y, 'merged_models',
          RandomForestRegressor(n_estimators=150, criterion="squared_error"), n=n_iteration)


def train_all(n_iteration=100):
    train_mask_forest(n_iteration)
    train_mask_nn(n_iteration)
    train_vert_forest(n_iteration)
    train_vert_nn(n_iteration)
    train_merge(n_iteration)


def get_min_model_error(models_dir):
    min_error = 10000
    for file in os.listdir(models_dir):
        file = file.replace('best', '')
        file = file.replace('.pickaim', '')
        error = min_error
        if file != '':
            error = float(file)
        if error < min_error:
            min_error = error
    return min_error


if __name__ == '__main__':
    # train_all(1000)
    # train_merge(2000)
    model = PointsDetector()
    pickle.dump(model, open('../points_model.pickaim', 'wb'))
