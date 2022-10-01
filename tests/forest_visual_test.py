import os
import pickle

import cv2

from forest_train import get_x


if __name__ == '__main__':
    with open('models/best.pickaim', 'rb') as f:
        model = pickle.load(f)
    for file in os.listdir('vert_masks'):
        img = cv2.imread(os.path.join('vert_masks', file))
        vert_img = cv2.imread(os.path.join('verts', file))
        points = model.predict([get_x(img)])[0]
        for i in range(0, len(points), 2):
            cv2.drawMarker(img, (int(points[i]), int(points[i + 1])), (0, 0, 255), cv2.MARKER_CROSS, 4, 2)
            cv2.drawMarker(vert_img, (int(points[i]), int(points[i + 1])), (0, 0, 255), cv2.MARKER_CROSS, 4, 2)
        cv2.imshow('res', img)
        cv2.imshow('res_vert', vert_img)
        cv2.waitKey(2500)
        cv2.imwrite(os.path.join('points_results', file), vert_img)

