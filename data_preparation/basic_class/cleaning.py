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
from pandas import Series


class DataClean(ABC):
    """
    数据清洗的标准类
    """

    @abstractmethod
    def __init__(self, x, mode, **params):
        """
        为所有数据清洗手段 统一输入数据的格式

        Args:
            x: Union[list, np.ndarray, pd.Series]
                Raw data
            mode: str
                数据清洗的算法名称
        """

        if isinstance(x, list) or isinstance(x, Series):
            x = np.asarray(x)
        elif not isinstance(x, np.ndarray):
            raise TypeError('Only support `list`, `np.ndarray`, `pd.Series`')
        self.x = x

        self.mode = mode
        self.params = params

    @abstractmethod
    def update(self, x, replace=True):
        """
        更新需要平滑的x
        """
        if replace:
            self.x = x
        else:
            x = np.concatenate((x, self.x), axis=0)
            self.x = x

    @abstractmethod
    def run(self):
        """
        运行 即对数据进行清洗
        """
        pass
