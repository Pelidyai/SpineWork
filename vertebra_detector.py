import time

import cv2
import numpy as np
import torch


def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * x + b))


def sigm_calc(vec, a, b):
    for j in range(len(vec)):
        vec[j] = sigmoid(j / len(vec), a, b)


def horizontal_gradient(in_image, space_window_width, a=15, b=10):
    img = in_image.copy()
    for i in range(img.shape[0]):
        vec1 = np.zeros((img.shape[1] - space_window_width) // 2)
        sigm_calc(vec1, a, b)
        vec3 = vec1[::-1]
        must_have = img.shape[1] - (img.shape[1] + space_window_width) // 2
        if len(vec1) < must_have:
            space_window_width += must_have - len(vec1)
        vec2 = ([1] * space_window_width)
        vec2 = np.array(vec2)
        vec2 = np.concatenate((vec2, vec3))
        vec1 = np.concatenate((vec1, vec2))
        line_copy = np.float64(img[i])
        line_copy *= vec1
        img[i] = np.uint8(line_copy)
    return img


def get_bias_img(img):
    left = img[:, 0:img.shape[1] // 2]
    right = img[:, img.shape[1] // 2:img.shape[1]]
    means = [np.mean(left), np.mean(right)]
    asymp_xs = [0, img.shape[1]]
    if abs(means[0] - means[1]) > 7:
        if np.argmax(means) == 0:
            asymp_xs[1] = img.shape[1] - right.shape[1] // 4
        else:
            asymp_xs[0] = left.shape[1] // 4
    return img[:, asymp_xs[0]:asymp_xs[1]], asymp_xs


def get_x_asymptote(img):
    asymp_xs = [0, 0]
    line = img[img.shape[0] - 1]
    for j in range(0, len(line) - 1):
        if line[j] == 0 and line[j + 1] != 0:
            asymp_xs[0] = j
            break
    for i in range(len(line) - 1, 1, -1):
        if line[i - 1] != 0 and line[i] == 0:
            asymp_xs[1] = i
            break
    return asymp_xs


def draw_box(img, box, sc):
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                  (0, 0, 255), 2)
    str = 'vertebra' + f'{round(sc, 3)}' + '%'
    cv2.putText(img, str, (int(box[0]) - 10, int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2)


def boxes_clean(boxes):
    out_boxes = []
    for i in range(len(boxes)):
        flag = True
        for j in range(len(boxes)):
            if boxes[i][0].item() < boxes[j][0].item() and boxes[i][1].item() < boxes[j][1].item() \
                    and boxes[i][2].item() > boxes[j][2].item() and boxes[i][3].item() > boxes[j][3].item():
                flag = False
                break
        if flag:
            out_boxes.append([boxes[i][0].item(), boxes[i][1].item(), boxes[i][2].item(), boxes[i][3].item()])
    return out_boxes


def get_spine(image):
    img = image.copy()
    orig = image.copy()
    img, bias_xs = get_bias_img(img)
    low = 150
    up = 255
    lower = np.array([low] * 3)
    upper = np.array([up] * 3)
    range1 = cv2.inRange(img, lower, upper)
    range1 = horizontal_gradient(range1, int(0.15 * img.shape[1]))
    low = 150
    up = 255
    lower = np.array([low] * 1)
    upper = np.array([up] * 1)
    range1 = cv2.inRange(range1, lower, upper)
    asymp_xs = get_x_asymptote(range1)
    img = orig[:, bias_xs[0] + asymp_xs[0]:asymp_xs[1] + bias_xs[0]]
    return img, bias_xs, asymp_xs


class VertebraDetector:
    def __init__(self, model_path, yolo_path):
        self.model = torch.hub.load(yolo_path, 'custom', path=model_path, source='local')

    def detect(self, img):
        spine, bias_xs, asymp_xs = get_spine(img)
        before = time.time()
        results = self.model(spine)
        print(time.time() - before)
        predictions = results.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        sc = []
        box = []
        if boxes != [[]]:
            for i in range(len(scores)):
                if scores[i].item() > 0.7:
                    sc.append(scores[i].item())
                    box.append(boxes[i])
        result = []
        result_box = []
        if box:
            box = boxes_clean(box)
            for i in range(len(box)):
                result.append(spine[int(box[i][1] - 10):int(box[i][3] + 10),
                              int(box[i][0] - 10):int(box[i][2] + 10)])
                result_box.append(box[i])
                # draw_box(spine, box[i], 0.0)
        return spine, result, result_box, bias_xs, asymp_xs
