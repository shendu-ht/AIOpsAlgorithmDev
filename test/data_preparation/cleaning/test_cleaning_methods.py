#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : test_cleaning_methods.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 9:43 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 9:43 下午 by shendu.ht  init
"""
import unittest

import numpy as np

from data_preparation.data_cleaning.remove import RemoveOutliers
from data_preparation.data_cleaning.smooth import DataSmooth
from data_preparation.data_cleaning.transform import DataTransform


class TestDataCleaning(unittest.TestCase):
    """
    测试数据清洗技巧
    """

    def test_remove(self):
        """
        测试剔除异常值下的诸多手段
        """
        x = np.array(range(1000))
        mode_list = ['RemoveNSigma']
        for mode in mode_list:
            ro = RemoveOutliers(x=x, mode=mode)
            ro.run()

    def test_smooth(self):
        """
        测试数据平滑处理的诸多手段
        """
        x = np.array(range(1000))
        mode_list = ['Diff', 'SG', 'MA']
        for mode in mode_list:
            dm = DataSmooth(x=x, mode=mode)
            dm.run()

    def test_transform(self):
        """
        测试数据格式转换下的各方法
        """
        x = np.array(range(1000))
        mode_list = ['SlideWindow']
        for mode in mode_list:
            dt = DataTransform(x=x, mode=mode)
            dt.run()


