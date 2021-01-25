#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : supervised.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 6:19 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 6:19 下午 by shendu.ht  init
"""
from abc import ABC, abstractmethod


class SupervisedModel(ABC):
    """
    Basic class for unsupervised learning.
    """

    @abstractmethod
    def __init__(self, x, y, **params):
        """

        Args:
            x: numpy.ndarray, shape=(n, n_features)
                Train input data
            y: numpy.ndarray, shape=(n, )
                Train output data
            **params: dict
                Other parameters
        """
        self.x = x
        self.y = y
        self.params = params

    @abstractmethod
    def fit(self):
        """
        Training process.
        """
        pass

    @abstractmethod
    def update(self, x_new, y_new):
        """
        Update training data.

        Args:
            x_new: numpy.ndarray, shape=(n, n_features)
                New input data we want to train
            y_new: numpy.ndarray, shape=(n, )
                New output data we want to train
        """
        pass

    @abstractmethod
    def predict(self, x_star):
        """
        Predict process.

        Args:
            x_star: numpy.ndarray, shape=(n, n_features)

        Returns:
            numpy.ndarray, shape=(n, )
                predict result for each sample.
        """
        pass

    @abstractmethod
    def evaluate(self, x_star, y_star):
        """
        Evaluate model precision.

        Args:
            x_star: numpy.ndarray, shape=(n, n_features)
                evaluate input data.
            y_star: numpy.ndarray, shape=(n, )
                evaluate output data.

        Returns:
            List[float]
                model precision. precision, recall, f1_score, MAE, MSE, MAPE, RMSE,...
        """
        pass

