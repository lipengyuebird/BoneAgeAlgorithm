#!venv/Scripts/python
# -*- coding: utf-8 -*-
# @file     : dataset.py
# @author   : lipengyu @CAICT
# @datetime : 2021/7/8 17:04 
# @software : PyCharm

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from util.config_loader import config
from util.annotation_reader import read_csv_line_2_kv
from torchvision import transforms


class BackBoneDataSet(Dataset):

    def __init__(self, transform: transforms.Compose = None, train_or_test: str = None):
        """
        load backbone network dataset

        :param transform: torchvision.transforms.Compose
        provide if loaded images need to transform

        :param train_or_test: 'train'(default), 'val' or 'test'
        indicates whether training dataset or testing dataset wil be used
        case will be ignored in this parameter

        """
        if not train_or_test or train_or_test.lower() not in ['train', 'val', 'test']:
            train_or_test = 'train'
        with open(config['dataset']['annotation'][train_or_test.lower() + '_img_name_csv'], 'r', encoding='utf-8') as f:
            self.raw_img_name_list = [line.strip() for line in f.readlines()[1:]]
            self.score_dict = read_csv_line_2_kv(config['dataset']['annotation']['score_csv'])
            self.transform = transform

    def __len__(self):
        return len(self.raw_img_name_list)

    def __getitem__(self, index) -> T_co:

        # convert image mode to RGB
        raw_img = Image.open(config['dataset']['img']['raw'] + self.raw_img_name_list[index]).convert("RGB")

        # ensure the image is in RGB mode
        assert raw_img.mode == 'RGB'

        if self.transform:
            raw_img = self.transform(raw_img)

        return raw_img, int(float(self.score_dict[self.raw_img_name_list[index]]))

    # def one_hot(self, label, num_classes):
    #     return np.array([0 if i != label else 1 for i in range(num_classes)])


class ROIDataSet(Dataset):

    def __init__(self, bone_name: str, transform: transforms.Compose = None, train_or_test: str = None):
        """
        load backbone network dataset

        :param bone_name: str

        options: RAO_GU, CHI_GU, DI_I_ZHANG_GU, DI_III_ZHANG_GU, DI_V_ZHANG_GU,
        DI_I_JIN_DUAN_ZHI_GU, DI_III_JIN_DUAN_ZHI_GU, DI_V_JIN_DUAN_ZHI_GU,
        DI_III_ZHONG_JIAN_ZHI_GU, DI_V_ZHONG_JIAN_ZHI_GU,
        DI_I_YUAN_DUAN_ZHI_GU, DI_III_YUAN_DUAN_ZHI_GU, DI_V_YUAN_DUAN_ZHI_GU,

        :param transform: torchvision.transforms.Compose
        provide if loaded images need to transform

        :param train_or_test: 'train'(default), 'val' or 'test'
        indicates whether training dataset or testing dataset wil be used
        case will be ignored in this parameter

        """

        assert bone_name in ['RAO_GU', 'CHI_GU', 'DI_I_ZHANG_GU', 'DI_III_ZHANG_GU', 'DI_V_ZHANG_GU',
                             'DI_I_JIN_DUAN_ZHI_GU', 'DI_III_JIN_DUAN_ZHI_GU', 'DI_V_JIN_DUAN_ZHI_GU',
                             'DI_III_ZHONG_JIAN_ZHI_GU', 'DI_V_ZHONG_JIAN_ZHI_GU',
                             'DI_I_YUAN_DUAN_ZHI_GU', 'DI_III_YUAN_DUAN_ZHI_GU', 'DI_V_YUAN_DUAN_ZHI_GU']

        self.transform = transform
        self.bone_name = bone_name

        if not train_or_test or train_or_test.lower() not in ['train', 'val', 'test']:
            train_or_test = 'train'
        with open(config['dataset']['annotation'][train_or_test.lower() + '_img_name_csv'], 'r', encoding='uft-8') as f:
            self.raw_img_name_list = [line.strip() for line in f.readlines()[1:]]

        self.name_img_path_dict = read_csv_line_2_kv(
            config['dataset']['annotation']['cropped_img_name_csv'], ['img_name', 'bone_name'], 'crop_img_path')
        self.name_label_dict = read_csv_line_2_kv(
            config['dataset']['annotation']['bone_label_csv'], ['img_name', 'bone_name'], 'label')

    def __len__(self):
        return len(self.raw_img_name_list)

    def __getitem__(self, index) -> T_co:

        # convert image mode to RGB
        cropped_img = Image.open(self.name_img_path_dict[(self.raw_img_name_list[index], self.bone_name)])\
            .convert("RGB")

        # ensure the image is in RGB mode
        assert cropped_img.mode == 'RGB'

        if self.transform:
            cropped_img = self.transform(cropped_img)

        return cropped_img, int(float(self.name_label_dict[(self.raw_img_name_list[index], self.bone_name)]))


if __name__ == '__main__':

    print(sorted(set([int(float(i)) for i in read_csv_line_2_kv(config['dataset']['annotation']['score_csv']).values()])))
