#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : test_supervised_algorithm.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 4:58 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 4:58 下午 by shendu.ht  init
"""
import random
import unittest
import warnings

from algorithm.supervised.gaussian_process_detector import GaussianProcessAD
from data_preparation.data_cleaning.transform import DataTransform
from example.algorithm.data_loader import data_loader


class TestSupervisedAD(unittest.TestCase):
    """
    测试基于监督学习的异常检测算法
    """

    def test_gaussian_process(self):
        """
        测试基于高斯过程的异常检测
        """
        warnings.filterwarnings("ignore")

        # 获取原始数据
        original_data = data_loader('jitter', 'ja_set_online.csv')

        # 原始数据转换成算法输入数据
        params = {'window': 15}
        dt = DataTransform(original_data, mode='SlideWindow', **params)
        x_train, y_train = dt.run()

        # 高斯过程
        index = [random.randint(0, x_train.shape[0] - 1) for _ in range(200)]
        gp_ad = GaussianProcessAD(x_train[index], y_train[index], n_iter=10)
        gp_ad.fit()





