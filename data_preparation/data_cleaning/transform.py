#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : transform.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 3:45 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 3:45 下午 by shendu.ht  init
"""
import numpy as np

from data_preparation.basic_class.cleaning import DataClean


class DataTransform(DataClean):
    """
    对数据进行重构
    """

    def __init__(self, x, mode='SlideWindow2D', **params):
        mode_dict = {'SlideWindow1D': self.slide_window_1d, 'SlideWindow2D': self.slide_window_2d}
        super().__init__(x=x, mode=mode, mode_dict=mode_dict, **params)

    def update_x(self, x, replace=True):
        super().update_x(x=x, replace=replace)

    def update_func(self, mode):
        super().update_func(mode=mode)

    def slide_window_1d(self, window=5):
        """
        数据划窗，1D -> 1D

        Parameters
        ----------
        window: int
            滑动窗口的大小

        Returns
        -------
        np.ndarray, shape=(self.x.shape[0] - window + 1, window)
            划窗处理后的时序输出

        """

        x_n = np.ones((self.x.shape[0] - window + 1, window))
        for i in range(x_n.shape[0]):
            x_n[i] = self.x[i:i + window]
        return x_n

    def slide_window_2d(self, window=15):
        """
        数据划窗，1D -> 2D

        Parameters
        ----------
        window: int
            滑动窗口的大小

        Returns
        ----------
        np.ndarray, shape=(self.x.shape[0] - window, window)
            数据划窗后的输入x
        np.ndarray, shape=(self.x.shape[0] - window,)
            数据划窗后的输出y
        """

        x_n = np.ones((self.x.shape[0] - window, window))
        y_n = np.ones((self.x.shape[0] - window,))
        y_n[:] = self.x[window:]

        for i in range(x_n.shape[0]):
            x_n[i] = self.x[i: i + window]
        return x_n, y_n

    def run(self):
        """
        运行 即对数据进行重构处理
        """
        return self.func(**self.params)
