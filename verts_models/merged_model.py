import pickle

from verts_models.mask_models import MaskForestModel, MaskNNModel
from verts_models.vert_models import VertForestModel, VertNNModel


class MergedModel:
    def __init__(self):
        self.mask_forest_model = MaskForestModel('../models_training/mask_forest_models/best.pickaim')
        self.mask_nn_model = MaskNNModel('../models_training/mask_forest_models/best.pickaim',
                                         '../models_training/mask_nn_models/best.pickaim')
        self.vert_forest_model = VertForestModel('../models_training/vert_forest_models/best.pickaim')
        self.vert_nn_model = VertNNModel('../models_training/vert_forest_models/best.pickaim',
                                         '../models_training/vert_nn_models/best.pickaim')
        self.model = pickle.load(open('../models_training/merged_models/best.pickaim', 'rb'))

    def get_x(self, mask_img, vert_img):
        result = []
        result.extend(self.mask_forest_model.predict(mask_img))
        result.extend(self.mask_nn_model.predict(mask_img))
        result.extend(self.vert_forest_model.predict(vert_img))
        result.extend(self.vert_nn_model.predict(vert_img))
        return result

    def predict(self, mask_img, vert_img):
        mas = [self.get_x(mask_img, vert_img)]
        return self.model.predict(mas)[0]
