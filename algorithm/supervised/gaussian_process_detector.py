#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : gaussian_process_detector.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 6:16 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 6:16 下午 by shendu.ht  init
"""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from algorithm.basic_class.supervised import SupervisedModel


class GaussianProcessAD(SupervisedModel):
    """
    Anomaly Detection based on gaussian process.
    """

    def __init__(self, x, y, kernel=None, optimizer=None, p=0.0001, alpha=1e-6, n_iter=10):
        """

        Args:
            x: np.ndarray, shape=(n, n_features)
                Train input data
            y: np.ndarray, shape=(n, )
                Train output data
            kernel: function
                Covariance function.
            optimizer: str
                optimizing the kernel's parameters
            p: float
                Statistical significance
            alpha: float
                Avoid zero standard var.
            n_iter: int
                The number if iterations for the optimizer to find the kernel's parameters.
        """
        super().__init__(x=x, y=y)
        if kernel is None:
            kernel = Matern(nu=2.5)
        if optimizer is None:
            optimizer = 'fmin_l_bfgs_b'
        self.kernel = kernel
        self.optimizer = optimizer
        self.ppf = norm.ppf(1 - p)
        self.alpha = alpha
        self.n_iter = n_iter

        self.gp = GaussianProcessRegressor(kernel=self.kernel, optimizer=self.optimizer, alpha=self.alpha,
                                           n_restarts_optimizer=self.n_iter, normalize_y=False)

    def fit(self):
        """
        Gaussian process training process.
        """
        self.gp.fit(self.x, self.y)

    def update(self, x_new, y_new):
        """
        Update Gaussian process training data.
        """
        x = np.concatenate((self.x, x_new), axis=0)
        y = np.concatenate((self.y, y_new), axis=0)
        self.gp.fit(x, y)

    def predict(self, x_star):
        """
        Predict y_star by Gaussian process

        Returns:
            np.ndarray, shape=(n, )
                predict y
            np.ndarray, shape=(n, )
                predict y lower line
            np.ndarray, shape=(n, )
                predict y upper line
        """
        y_mean, y_cov = self.gp.predict(x_star, return_cov=True)
        std = np.sqrt(np.diag(y_cov))
        lower = y_mean - std * self.ppf
        upper = y_mean + std * self.ppf
        return y_mean, lower, upper

    def evaluate(self, x_star, y_star):
        """
        Evaluate gaussian process model's precision

        Returns:
            np.ndarray, shape=(n, )
                Absolute percentage error
            float
                MAPE
        """
        y_mean, lower, upper = self.predict(x_star)
        ape = np.abs((y_mean - y_star) / (y_star + self.alpha))
        return ape, np.average(ape)
