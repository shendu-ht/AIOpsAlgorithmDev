#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : cleaning.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 10:54 上午
    Description : description what the main function of this file
    Change Activity :
            version0 : 10:54 上午 by shendu.ht  init
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class DataClean(ABC):
    """
    数据清洗的标准类
    """

    @abstractmethod
    def __init__(self, x, mode, mode_dict, **params):
        """
        为所有数据清洗手段 统一输入数据的格式

        Args:
            x: Union[list, np.ndarray, pd.Series]
                Raw data
            mode: str
                数据清洗的算法名称
        """

        if isinstance(x, list) or isinstance(x, pd.Series):
            x = np.asarray(x)
        elif not isinstance(x, np.ndarray):
            raise TypeError('Only support `list`, `np.ndarray`, `pd.Series`')
        self.x = x

        self.mode = mode
        self.mode_dict = mode_dict
        self.func = self.mode_dict[self.mode]

        self.params = params

    @abstractmethod
    def update_x(self, x, replace=True):
        """
        更新需要清洗的x
        """

        if isinstance(x, list) or isinstance(x, pd.Series):
            x = np.asarray(x)
        elif not isinstance(x, np.ndarray):
            raise TypeError('Only support `list`, `np.ndarray`, `pd.Series`')

        if replace:
            self.x = x
        else:
            self.x = np.concatenate((x, self.x), axis=0)

    @abstractmethod
    def update_func(self, mode):
        self.func = self.mode_dict[mode]

    @abstractmethod
    def run(self):
        """
        运行 即对数据进行清洗
        """
        pass
