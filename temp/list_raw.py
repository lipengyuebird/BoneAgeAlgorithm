#!venv/Scripts/python
# -*- coding: utf-8 -*-
# @file     : list_raw.py
# @author   : lipengyu @CAICT
# @datetime : 2021/7/12 17:14 
# @software : PyCharm

import os
import csv
# from util.config_loader import config

if __name__ == '__main__':

    unaugmented = [i for i in os.listdir('C:\\BoneAgeAssessment\\Route2\\dataset\\raw_img\\') if '.jpg' in i and len(i) >= 3 and i.replace('.jpg', '')[-2] != '_' and i.replace('.jpg', '')[-3] != '_']

    multiple_dict = {}

    with open('C:\\BoneAgeAssessment\\Route1\\dataset\\annos\\special_data.csv', 'r') as f:

        for line in csv.reader(f):
            try:
                multiple_dict[line[0]] = int(line[3])
            except ValueError:
                pass

    with open('C:\\BoneAgeAssessment\\Route2\\dataset\\annos\\test_img_name.csv') as f:

        test = [line.strip() for line in f.readlines()[1:]]

        for img_name in list(set(unaugmented) - set(test)):

            multiple = 20

            if img_name in multiple_dict.keys():

                multiple = multiple_dict[img_name]

            for i in range(multiple):
                print(f'正在复制 {img_name} {i}')
                if i == 0:
                    continue
                os.system(f'copy C:\\BoneAgeAssessment\\Route2\\dataset\\raw_img\\{img_name} C:\\BoneAgeAssessment\\Route2\\dataset\\raw_img\\{img_name.replace(".jpg", "_" + str(i) + ".jpg")}')