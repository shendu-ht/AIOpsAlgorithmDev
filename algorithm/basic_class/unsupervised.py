#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : unsupervised.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 5:37 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 5:37 下午 by shendu.ht  init
"""
from abc import ABC, abstractmethod


class UnsupervisedStats(ABC):
    """
    Basic class for unsupervised learning.
    """

    @abstractmethod
    def __init__(self, x, **params):
        """

        Args:
            x: numpy.ndarray, shape=(n, n_features)
                Train data
            **params: dict
                Other parameters
        """
        self.x = x
        self.params = params

    @abstractmethod
    def fit(self):
        """
        training process.
        """
        pass

    @abstractmethod
    def predict(self, x_star):
        """
        predict process.

        Args:
            x_star: numpy.ndarray, shape=(n, n_features)

        Returns:
            numpy.ndarray, shape=(n, 1)
                predict result for each sample. Union[bool, float, int]
        """
        pass

    @abstractmethod
    def evaluate(self, x_star, y_star):
        """
        evaluate model precision.

        Args:
            x_star: numpy.ndarray, shape=(n, n_features)
            y_star: numpy.ndarray, shape=(n, )

        Returns:
            List[float]
                model precision. precision, recall, f1_score, MAE, MSE, MAPE, RMSE,...
        """
        pass
