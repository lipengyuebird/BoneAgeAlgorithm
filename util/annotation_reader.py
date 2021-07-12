#!venv/Scripts/python
# -*- coding: utf-8 -*-
# @file     : annotation_reader.py
# @author   : lipengyu @CAICT
# @datetime : 2021/7/9 10:48 
# @software : PyCharm

import csv


def read_csv_line_2_kv(csv_file_path,
                       key_columns: list or tuple or str = None,
                       value_columns: list or tuple or str = None):

    """
    convert a csv file to dict by transforming a line into a key-value pair \n

    :param csv_file_path: str

    :param key_columns: str, list, or tuple
    title of column(s) used as key

    :param value_columns: str, list, or tuple
    title of column(s) used as value \n
    key_columns and value_columns should be both provided or absent,
    providing only one of them is not allowed

    :return: result dict

    eg. for a csv file like \n

    col1, col2, col3 \n
    a1, a2, a3 \n

    read_csv_line_2_kv('./filename', ['col1', 'col3'], ['col2']) returns
    {('a1', 'a3'): 'a2'}
    """

    # ensure both key columns and value columns are provided or absent
    assert (key_columns is None) == (value_columns is None)

    kv_dict = {}

    with open(csv_file_path, 'r', encoding='utf-8') as f:
        title_list = f.readline().strip().split(',')

        if not key_columns:
            key_columns = title_list[0]
            value_columns = title_list[1]

        # initialize the title index dict
        # eg.: col1, col2, col3 -> {'col1': 0, 'col2': 1, 'col3': 2}
        title_index_dict = {}
        for i in range(len(title_list)):
            title_index_dict[title_list[i]] = i

        for row in csv.reader(f):
            if isinstance(key_columns, str):
                key = row[title_index_dict[key_columns]].strip()
            elif isinstance(key_columns, (list, tuple)):
                key = tuple([row[title_index_dict[i]].strip() for i in key_columns])

            if isinstance(value_columns, str):
                value = row[title_index_dict[value_columns]].strip()
            elif isinstance(key_columns, (list, tuple)):
                value = [row[title_index_dict[i]].strip() for i in value_columns]
            kv_dict[key] = value

    return kv_dict


# if __name__ == '__main__':
#     print(read_csv_line_2_kv('C:\\BoneAgeAssessment\\Route1\\dataset\\annos\\bone_label.csv',
#                              key_columns=['img_name', 'bone_name'],
#                              value_columns='label'))
