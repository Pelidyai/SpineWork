import pickle

from vertebra_detector import VertebraDetector

if __name__ == '__main__':
    model_path = 'D:/PickAim/Projects/SpineWork/yolov5/runs/train/exp10/weights/best.pt'
    yolo_path = 'D:/PickAim/Projects/SpineWork/yolov5'
    detector = VertebraDetector(model_path, yolo_path)
