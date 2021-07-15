#!venv/Scripts/python
# -*- coding: utf-8 -*-
# @file     : aa.py
# @author   : lipengyu @CAICT
# @datetime : 2021/7/12 17:53 
# @software : PyCharm



import os
import csv
from util.config_loader import config

if __name__ == '__main__':

    # unaugmented = [i for i in os.listdir(config['dataset']['img']['raw']) if '.jpg' in i and len(i) >= 3 and i.replace('.jpg', '')[-2] != '_' and i.replace('.jpg', '')[-3] != '_']

    multiple_dict = {}

    # with open('C:\\BoneAgeAssessment\\Route1\\dataset\\annos\\special_data.csv', 'r') as f:
    #
    #     for line in csv.reader(f):
    #         try:
    #             multiple_dict[line[0]] = int(line[3])
    #         except ValueError:
    #             pass

    with open(config['dataset']['annotation']['val_img_name_csv']) as f:

        # score_img_list = []

        with open('C:\\BoneAgeAssessment\\Route2\\dataset\\annos\\score.csv', 'r', encoding='utf-8') as f1:
            score_img_list = [line.strip().split(',')[0] for line in f1.readlines()[1:]]

        train = set([line.strip() for line in f.readlines()[1:]])
        train = train- set(score_img_list)
        # with open('C:\\BoneAgeAssessment\\Route2\\dataset\\annos\\val_img_name11.csv', 'w', encoding='utf-8', newline='\n') as f1:
        #
        #     for line in sorted(list(train)):
        #         f1.write(line + '\n')

        print(train)