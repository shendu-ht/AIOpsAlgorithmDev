#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : augmentation.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 11:31 上午
    Description : description what the main function of this file
    Change Activity :
            version0 : 11:31 上午 by shendu.ht  init
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class DataAugmentation(ABC):
    """
    数据增强的标准类
    """

    @abstractmethod
    def __init__(self, x, mode, mode_dict, **params):
        """

        Parameters
        ----------
        x: list, np.ndarray, pd.Series
            待进行数据增强的原始数据
        mode: str
            数据增强算法名称
        params: dict
            实现类的额外参数输入
        """

        if isinstance(x, list) or isinstance(x, pd.Series):
            x = np.asarray(x)
        elif not isinstance(x, np.ndarray):
            raise TypeError('Only support `list`, `np.ndarray`, `pd.Series`')
        self.x = x

        self.mode = mode
        self.mode_dict = mode_dict
        self.func = self.mode_dict[mode]

        self.params = params

    @abstractmethod
    def update(self, x, replace=True):
        """
        更新需要增强的数据x

        Parameters
        ----------
        x: list, np.ndarray, pd.Series
            待进行更新的数据x
        replace:
            若为True，进行数据替换，若为false，进行数据补充
        """

        if isinstance(x, list) or isinstance(x, pd.Series):
            x = np.asarray(x)
        elif not isinstance(x, np.ndarray):
            raise TypeError('Only support `list`, `np.ndarray`, `pd.Series`')

        if replace is True:
            self.x = x
        else:
            self.x = np.concatenate((x, self.x), axis=0)

    @abstractmethod
    def run(self):
        """
        运行，即对数据进行增强
        """
        pass
