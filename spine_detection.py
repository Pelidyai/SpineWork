import cv2
import numpy as np
import torch
import os
import argparse
import sys


def show(win_name, img, scale=0.3):
    im = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                    interpolation=cv2.INTER_LINEAR)
    cv2.imshow(win_name, im)


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
    cv2.rectangle(img, (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())),
                  (0, 0, 255), 2)
    str = 'vertebra' + f'{round(sc, 3)}' + '%'
    cv2.putText(img, str, (int(box[0].item()) - 10, int(box[1].item()) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
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
            out_boxes.append(boxes[i])
    return out_boxes


def detect(img):
    model_path = 'D:/PickAim/Projects/SpineWork/yolov5/runs/train/exp10/weights/best.pt'
    yolo_path = 'D:/PickAim/Projects/SpineWork/yolov5'
    model = torch.hub.load(yolo_path, 'custom', path=model_path, source='local')
    results = model(img)
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
    if box:
        box = boxes_clean(box)
        for i in range(len(box)):
            draw_box(img, box[i], sc[i])
    return img


def get_spine(image):
    img = image.copy()
    orig = image.copy()
    img, bias_xs = get_bias_img(img)
    low = 150
    up = 255
    lower = np.array([low] * 3)
    upper = np.array([up] * 3)
    range1 = cv2.inRange(img, lower, upper)
    range1 = horizontal_gradient(range1, int(0.05 * img.shape[1]))
    low = 150
    up = 255
    lower = np.array([low] * 1)
    upper = np.array([up] * 1)
    range1 = cv2.inRange(range1, lower, upper)
    asymp_xs = get_x_asymptote(range1)
    img = orig[:, bias_xs[0] + asymp_xs[0]:asymp_xs[1] + bias_xs[0]]
    return img, bias_xs, asymp_xs


def create_parser():
    r = argparse.ArgumentParser()
    r.add_argument('-i', '--image')
    return r


if __name__ == '__main__':
    # arg_parser = create_parser()
    # namespace = arg_parser.parse_args(sys.argv[1:])
    # img_path = namespace.image
    # filename = str.split(img_path, os.sep)[-1]
    # filename = str.split(filename, '.')[0]
    img_path = '1.JPG'
    image = cv2.imread(img_path)
    inner, bias, asymp = get_spine(image)
    show('inner', inner)
    inner = detect(inner)
    image[:, bias[0] + asymp[0]:asymp[1] + bias[0]] = inner
    show('res', image)
    # cv2.imwrite(os.path.join('res', 'img', filename + '.jpg'), image)
    cv2.waitKey(0)
