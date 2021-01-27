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
import numpy


def slide_window(x, window=10):
    """
    Transform 1D series data to 2D matrix, by rolling.

    Args:
        x: numpy.ndarray, shape=(n, )
            Input series data
        window: int
            Window size

    Returns:
        numpy.ndarray, shape=(n - window + 1, n_features)
            x for training
        numpy.ndarray, shape=(n - window + 1, )
            y for training
    """

    x_n = numpy.ones((x.shape[0] - window, window))
    y_n = numpy.ones((x.shape[0] - window, ))
    y_n[:] = x[window:]

    for i in range(x_n.shape[0]):
        x_n[i] = x[i: i + window]
    return x_n, y_n
