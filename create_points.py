import os

import cv2
import numpy as np

mas = []


# mouse callback function
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        mas.append((x, y))


# Создание изображений и окон и привязка окон к функциям обратного вызова
files = os.listdir('verts')
for file in files:
    img = cv2.imread(os.path.join('verts', file))
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while (1):
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    with open(os.path.join('points', file.split('.')[0] + '.txt'), 'w') as f:
        for point in mas:
            f.write(str(point[0]) + " " + str(point[1]) + "\n")
    mas = []
cv2.destroyAllWindows()