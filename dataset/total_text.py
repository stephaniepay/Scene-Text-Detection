import scipy.io as io
import numpy as np
import os

from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance

class TotalText(TextDataset):

    def __init__(self, data_root, ignore_list=None, is_training=True, transform=None, data_custom=False):
        super().__init__(transform)
        self.data_root = data_root
        self.is_training = is_training

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

#         self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
#         self.annotation_root = os.path.join(data_root, 'gt', 'Train' if is_training else 'Test')
#         self.image_list = os.listdir(self.image_root)

#         self.image_list = list(filter(lambda img: img.replace('.jpg', '').replace('.JPG', '') not in ignore_list, self.image_list))

#         self.annotation_list = ['poly_gt_{}.mat'.format(img_name.replace('.jpg', '').replace('.JPG', '')) for img_name in self.image_list]




        self.data_custom = data_custom
        if not data_custom:
            self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
            self.annotation_root = os.path.join(data_root, 'gt', 'Train' if is_training else 'Test')
            self.image_list = os.listdir(self.image_root)
            self.image_list = list(filter(lambda img: img.replace('.jpg', '').replace('.JPG', '') not in ignore_list, self.image_list))
            self.annotation_list = ['poly_gt_{}.mat'.format(img_name.replace('.jpg', '').replace('.JPG', '')) for img_name in self.image_list]
        else:
            self.image_root = os.path.join(data_root, 'train' if is_training else 'test', 'images')
            self.boxes_root = os.path.join(data_root, 'train' if is_training else 'test', 'boxes')
            self.text_root = os.path.join(data_root, 'train' if is_training else 'test', 'text')
            self.image_list = os.listdir(self.image_root)
            self.boxes_list = [img_name.replace('.jpeg', '.txt') for img_name in self.image_list]
            self.text_list = [img_name.replace('.jpeg', '.txt') for img_name in self.image_list]


    def parse_txt(self, boxes_path, text_path):
        """
        .txt file parser
        :param txt_path: (str), txt file path
        :return: (list), TextInstance
        """
        polygon = []
        boxes = np.genfromtxt(boxes_path,delimiter=',').astype(np.int32)
        text = [item.strip() for item in open(text_path).readlines()]
        assert len(boxes) == len(text), 'Length must equal'

        for box, t in zip(boxes, text):
            t = np.str_(t)
            ori = np.str_('m')
            pts = np.asarray(box).reshape((4, 2)).astype(np.int32)
            polygon.append(TextInstance(pts, ori, text))
        return polygon







    def parse_mat(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path)
        polygons = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            ori = cell[5][0] if len(cell[5]) > 0 else 'c'

            if len(x) < 4:  # too few points
                continue
            pts = np.stack([x, y]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))

        return polygons

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)

        # Read annotation
#         annotation_id = self.annotation_list[item]
#         annotation_path = os.path.join(self.annotation_root, annotation_id)
#         polygons = self.parse_mat(annotation_path)

#         for i, polygon in enumerate(polygons):
#             if polygon.text != '#':
#                 polygon.find_bottom_and_sideline()

        if not self.data_custom:
            annotation_id = self.annotation_list[item]
            annotation_path = os.path.join(self.annotation_root, annotation_id)
            polygons = self.parse_mat(annotation_path)
        else:
            polygons = self.parse_txt(os.path.join(self.boxes_root, self.boxes_list[item]), \
                                      os.path.join(self.boxes_root, self.text_list[item]))

        return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    import os
    from util.augmentation import BaseTransform, Augmentation

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=512, mean=means, std=stds
    )

    trainset = TotalText(
        data_root='data/total-text',
        # ignore_list='./ignore_list.txt',
        is_training=True,
        transform=transform
    )

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]

    for idx in range(0, len(trainset)):
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[idx]
        print(idx, img.shape)