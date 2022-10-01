import pickle
import random

import cv2

from mask_creation import show
from vertebra_detector import VertebraDetector

if __name__ == '__main__':
    # cur_dir = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(cur_dir, 'yolov5/runs/train/exp10/weights/best.pt')
    # yolo_path = os.path.join(cur_dir, 'yolov5')
    model_path = 'D:/PickAim/Projects/SpineWork/yolov5/runs/train/exp10/weights/best.pt'
    yolo_path = 'D:/PickAim/Projects/SpineWork/yolov5'
    verts_detector = VertebraDetector(model_path, yolo_path)
    points_detector = pickle.load(open('../points_model.pickaim', 'rb'))
    # points_detector = PointsDetector()
    img = cv2.imread('../1.JPG')
    spine, verts, boxes, bias, asymp = verts_detector.detect(img)
    verts = verts.copy()
    points = points_detector.get_points(verts)
    delta = 255//len(points)
    colors = {
        True: (0, 0, 255),
        False: (255, 0, 0)
    }
    is_switch = True
    for i in range(len(points)):
        color = (int(random.Random().random() * 255),
                 int(random.Random().random() * 255),
                 int(random.Random().random() * 255))
        for j in range(0, len(points[i]), 2):
            cv2.drawMarker(verts[i], (int(points[i][j]),
                                      int(points[i][j + 1])), colors[True], cv2.MARKER_CROSS, 6, 4)
        # cv2.imwrite(f'{i}.jpg', verts[i])
        is_switch = not is_switch
        # show('spine', verts[i], 1.2)
        # cv2.waitKey(0)
    for i in range(len(boxes)):
        vert = verts[i]
        spine[int(boxes[i][1] - 10):int(boxes[i][3] + 10), int(boxes[i][0] - 10):int(boxes[i][2] + 10)] \
            = vert[:int(boxes[i][3] + 10), :int(boxes[i][2] + 10)]
    # show('spine', spine)
    # cv2.imwrite('res.jpg', spine)
    img[:, bias[0] + asymp[0]:asymp[1] + bias[0]] = spine
    show('result', img)
    cv2.imwrite('res.jpg', img)
    cv2.waitKey(0)
