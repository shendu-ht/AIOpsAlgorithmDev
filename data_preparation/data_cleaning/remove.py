#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : remove.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 8:43 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 8:43 下午 by shendu.ht  init
"""
from copy import copy

from data_preparation.basic_class.cleaning import DataClean
from data_preparation.constant import Constant


class RemoveOutliers(DataClean):
    """
    剔除数据中的异常值
    """

    def __init__(self, x, mode='RemoveNSigma', **params):
        super().__init__(x=x, mode=mode, **params)

        mode_dict = {'RemoveNSigma': self.remove_n_sigma_outlier}
        self.func = mode_dict[mode]

    def update(self, x, replace=True):
        super().update(x=x, replace=replace)

    def remove_n_sigma_outlier(self, n_iter=3):
        """
        循环迭代剔除数据中的异常值
        """

        stop_loop = False
        x_copy = copy(self.x)

        # 循环剔除异常值，直至数据中不存在异常值
        while n_iter > 0 and not stop_loop:
            upper = x_copy.mean() + Constant.N_SIGMA * x_copy.std()
            lower = x_copy.mean() - Constant.N_SIGMA * x_copy.std()
            x_copy_size = x_copy.size
            x_copy = x_copy[(x_copy <= upper) * (x_copy >= lower)]

            if x_copy.size == x_copy_size:
                break
            n_iter -= 1

        return x_copy

    def run(self):
        """
        运行函数，剔除异常值，返回没有异常值的数组
        """
        return self.func(**self.params)
