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
import unittest

from algorithm.supervised.gaussian_process_detector import GaussianProcessAD
from data_preparation.cleaning.transform import slide_window
from example.algorithm.data_loader import data_loader


class TestSupervisedAD(unittest.TestCase):

    def test_gaussian_process(self):
        original_data = data_loader()
        window_list = [5, 10, 15, 20]
        test_len = 360

        for window in window_list:
            x_n, y_n = slide_window(original_data, window)
            x_train, y_train = x_n[:-test_len, :], y_n[:-test_len]
            x_test, y_test = x_n[-test_len:, :], y_n[-test_len:]

            gp_ad = GaussianProcessAD(x_train, y_train)
            gp_ad.fit()
            ape, mape = gp_ad.evaluate(x_test, y_test)
            print(mape)



