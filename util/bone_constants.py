#!venv/Scripts/python
# -*- coding: utf-8 -*-
# @file     : bone_constants.py
# @author   : lipengyu @CAICT
# @datetime : 2021/7/9 16:33 
# @software : PyCharm

class BoneConstants:
    """
    bone related constants
    """

    # 桡骨 Ulna
    ULNA = -1
    RAO = -1

    # 尺骨 Radius
    RADIUS = 0
    CHI = 0

    # 拇指 Thumb finger
    I: int = 1
    THUMB_FINGER = 1

    # 食指 Index finger
    II = 2
    INDEX_FINGER = 2

    # 中指 Middle finger
    III = 3
    MIDDLE_FINGER = 3

    # 无名指 Ring finger
    IV = 4
    RING_FINGER = 4

    # 小拇指 Little finger
    V = 5
    LITTLE_FINGER = 5

    class End:

        # 掌骨 Meta carpel
        META_CARPEL = 0

        # 近端 Proximal
        PROXIMAL = 1

        # 中端 Middle
        MIDDLE = 2

        # 远端 Distal
        DISTAL = 3

    @staticmethod
    def get_index(c1, c2):

        if c1 == -1:
            return 1
        elif c1 == 0:
            return 4
        else:
            return c1 * 4 + c2 + 14

    # @staticmethod
    # def get_identifier(index):
    #
    #     return ['RAO_GU', 'CHI_GU', 'DI_I_ZHANG_GU', 'DI_III_ZHANG_GU', 'DI_V_ZHANG_GU',
    #     'DI_I_JIN_DUAN_ZHI_GU', 'DI_III_JIN_DUAN_ZHI_GU', 'DI_V_JIN_DUAN_ZHI_GU',
    #     'DI_III_ZHONG_JIAN_ZHI_GU', 'DI_V_ZHONG_JIAN_ZHI_GU',
    #     'DI_I_YUAN_DUAN_ZHI_GU', 'DI_III_YUAN_DUAN_ZHI_GU', 'DI_V_YUAN_DUAN_ZHI_GU']
