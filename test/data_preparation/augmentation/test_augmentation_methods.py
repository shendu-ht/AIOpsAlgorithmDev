#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : test_augmentation_methods.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 5:23 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 5:23 下午 by shendu.ht  init
"""
import unittest

import numpy as np

from data_preparation.data_augmentation.spectral_residual import SR


class TestDataCleaning(unittest.TestCase):
    """
    测试数据增强的手段
    """

    def test_spectral_residual(self):
        """
        测试基于sr的数据增强手段
        """
        x = np.array(range(1000))
        mode_list = ['SR']
        for mode in mode_list:
            sr = SR(x=x, mode=mode)
            sr.run()
