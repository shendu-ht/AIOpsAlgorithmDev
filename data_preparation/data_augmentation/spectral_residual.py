#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : spectral_residual.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 11:51 上午
    Description : description what the main function of this file
    Change Activity :
            version0 : 11:51 上午 by shendu.ht  init
"""
import numpy as np

from data_preparation.basic_class.augmentation import DataAugmentation
from data_preparation.data_cleaning.smooth import DataSmooth


class SR(DataAugmentation):
    """
    Spectral Residual 光谱残差，用于对时序数据进行低频成分抽取
    """

    def __init__(self, x, mode='SR', **params):
        mode_dict = {'SR': self.transform_by_sr}
        super().__init__(x=x, mode=mode, mode_dict=mode_dict, **params)

    def update(self, x, replace=True):
        super().update(x=x, replace=replace)

    def transform_by_sr(self, window=5):
        """
        使用SR对原始数据进行转换

        Parameters
        ----------
        window: int
            移动平均的窗口长度
        """

        freq = np.fft.fft(self.x)
        mag = np.sqrt(freq.real ** 2 + freq.imag ** 2)

        mag_log = np.log(mag)
        params = {'method': 'avg', 'window': window}
        dm = DataSmooth(mag_log, mode='MA', **params)
        mag_smooth = dm.run()
        sr = np.exp(mag_log - mag_smooth)

        freq.real = freq.real * sr / mag
        freq.imag = freq.imag * sr / mag
        return np.fft.ifft(freq)

    def sr(self):
        """
        输入x的spectral residual
        """
        x_n = self.transform_by_sr(window=5)
        return np.sqrt(x_n.real ** 2 + x_n.imag ** 2)

    def run(self):
        """
        运行 即利用光谱残差对数据进行处理
        """
        return self.func(**self.params)
