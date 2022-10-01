import os
from models.merged_model import MergedModel
import pickle

import cv2


if __name__ == '__main__':
    # merged_model = pickle.load(open('model.pickaim', 'rb'))
    merged_model = MergedModel()
    for file in os.listdir('../vert_masks'):
        mask_img = cv2.imread(os.path.join('../vert_masks', file))
        vert_img = cv2.imread(os.path.join('../verts', file))
        merged_points = merged_model.predict(mask_img, vert_img)
        for i in range(0, len(merged_points), 2):
            # cv2.drawMarker(mask_img, (int(forest_mask_points[i]),
            #                           int(forest_mask_points[i + 1])), (0, 0, 255), cv2.MARKER_CROSS, 4, 2)
            # cv2.drawMarker(mask_img, (int(nn_mask_points[i]),
            #                           int(nn_mask_points[i + 1])), (255, 0, 255), cv2.MARKER_CROSS, 4, 2)
            # cv2.drawMarker(vert_img, (int(forest_vert_points[i]),
            #                           int(forest_vert_points[i + 1])), (0, 0, 255), cv2.MARKER_CROSS, 4, 2)
            # cv2.drawMarker(vert_img, (int(nn_vert_points[i]),
            #                           int(nn_vert_points[i + 1])), (255, 0, 255), cv2.MARKER_CROSS, 4, 2)
            cv2.drawMarker(vert_img, (int(merged_points[i]),
                                      int(merged_points[i + 1])), (0, 255, 0), cv2.MARKER_CROSS, 4, 2)
        cv2.imshow('res', mask_img)
        cv2.imshow('res_vert', vert_img)
        cv2.waitKey(100)
        # cv2.imwrite(os.path.join('../points_merged', file), vert_img)
    pickle.dump(merged_model, open('../merged_model.pickaim', 'wb'))

