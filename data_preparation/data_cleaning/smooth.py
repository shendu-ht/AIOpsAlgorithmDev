#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : smooth.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 4:26 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 4:26 下午 by shendu.ht  init
"""
from copy import copy

import numpy as np

from data_preparation.basic_class.cleaning import DataClean
from data_preparation.constant import Constant
from data_preparation.data_cleaning.remove import RemoveOutliers


class DataSmooth(DataClean):
    """
    数据平滑
    """

    def __init__(self, x, mode='Diff', **params):
        super().__init__(x=x, mode=mode, **params)

        mode_dict = {'Diff': self.diff_smooth, 'SG': self.savitzky_golay, 'MA': self.move_avg}
        self.func = mode_dict[self.mode]

    def update(self, x, replace=True):
        super().update(x=x, replace=replace)

    def diff_smooth(self, method='N-Sigma'):
        """
        通过对原始数据做差分，识别异常点，并通过异常前和异常后的数据对其进行平滑处理
        """

        diff_x = np.diff(self.x)
        if method == 'N-Sigma':
            # 因为极值影响，正常波动的标准差易受到影响，需剔除极值
            diff_x_copy = copy(diff_x)
            ro = RemoveOutliers(diff_x_copy)
            diff_x_copy = ro.run()

            # 基于剔除极值后的数据，计算新的上下界
            upper = diff_x.mean() + Constant.N_SIGMA * diff_x_copy.std()
            lower = diff_x.mean() - Constant.N_SIGMA * diff_x_copy.std()

        elif method == 'Boxplot':
            q_75, q_25 = np.percentile(diff_x, [75, 25])
            upper = q_75 + Constant.BOXPLOT_IQR * (q_75 - q_25)
            lower = q_25 - Constant.BOXPLOT_IQR * (q_75 - q_25)
        else:
            raise TypeError('Only support `N-sigma`, `Boxplot`')

        # 超过upper，低于lower的value为True，否则为False
        index = ~((diff_x <= upper) * (diff_x >= lower))

        smoothed_x = self.x
        # for i in range(len(index)):
        i = 0
        while i < len(index):
            j = 0

            # 当出现连续的抖动异常时，可能存在连续两个异常状态类似
            # index[i + j], index[i + j + 1] 连续两个状态有一个异常，就会继续查找
            while (i + j < len(index) and index[i + j]) or (i + j + 1 < len(index) and index[i + j + 1]):
                j += 1

            if j > 0:
                # 考虑到数据异常时可能存在波动，前几分钟无法被平滑，因此在平滑时的数据选取，index会进一步调整
                start_index = max(0, i - 5)
                end_index = min(i + j + 5, len(self.x) - 1)
                smoothed_x[i + 1: i + j + 1] = np.linspace(self.x[start_index], self.x[end_index], j)
            i += j + 1
        return smoothed_x

    def savitzky_golay(self):
        """
        待添加。savitzky_golay滤波法，对窗口内数据进行加权滤波，加权权重通过给定的高阶多项式进行最小二乘法拟合所得
        """
        pass

    def move_avg(self, method='exp'):
        """
        待添加。移动平均法，既可以选择`exp`添加decay衰减系数，对移动窗口内的数据进行加权，也可选择`avg`不加权重
        """
        pass

    def run(self):
        """
        运行 即对数据进行平滑处理
        """
        return self.func(**self.params)
