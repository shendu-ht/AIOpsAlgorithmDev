#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : temp.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 5:06 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 5:06 下午 by shendu.ht  init
"""
import random
import warnings
from copy import copy

import numpy
from matplotlib import pyplot

from algorithm.supervised.gaussian_process_detector import GaussianProcessAD
from data_preparation.data_cleaning.smooth import DataSmooth
from data_preparation.data_cleaning.transform import DataTransform
from example.algorithm.data_loader import data_loader


def main():
    original_data = data_loader('jitter', 'ja_set_online.csv')
    window_list = [15]  # [5, 10, 15, 20, 30]
    smooth_data = copy(original_data)
    ds = DataSmooth(smooth_data)
    smooth_data = ds.run()

    # print(','.join(['window_size', 'sample size', 'test round', 'mape']))

    for window in window_list:
        params = {'window': window}
        dt = DataTransform(smooth_data, mode='SlideWindow', **params)
        x_train, y_train = dt.run()
        dt.update(original_data, replace=True)
        x_test, y_test = dt.run()

        sample_size_list = [200]  # [100, 200]
        for sample_size in sample_size_list:
            # for i in range(100):
            index = [random.randint(0, x_train.shape[0] - 1) for _ in range(sample_size)]
            gp_ad = GaussianProcessAD(x_train[index], y_train[index], n_iter=10)
            gp_ad.fit()
            y_mean, lower, higher = gp_ad.predict(x_train)

            index = ~((y_test > lower) * (y_test < higher))

            x = numpy.array(range(0, len(y_test)))

            pyplot.scatter(x[index], y_test[index], color='r')
            pyplot.plot(x, y_test, color='b')

            pyplot.fill_between(x, lower, higher, alpha=0.2)
            pyplot.show()
    return


def mock_data():
    original_data = data_loader('jitter', 'ja_set_online.csv')
    smooth_data = copy(original_data)
    ds = DataSmooth(smooth_data)
    smooth_data = ds.run()
    window = 15
    params = {'window': window}
    dt = DataTransform(smooth_data, mode='SlideWindow', **params)
    x_train, y_train = dt.run()
    dt.update(original_data, replace=True)
    x_test, y_test = dt.run()

    # pyplot.plot(original_data)
    # pyplot.show()

    # abnormal = False
    # y_i = None
    # y_mean = []
    lower_list = []
    higher_list = []

    start = 2730 - window
    end = 2880 - window
    y_target = y_test[start: end]
    y_smooth = y_train[start: end]

    index = []

    for i in range(start, end):
        x_train_i = x_train[i - 200:i]
        y_train_i = y_train[i - 200:i]

        gp_ad = GaussianProcessAD(x_train_i, y_train_i, p=0.0001, n_iter=10)
        gp_ad.fit()

        x_i = numpy.array([x_train[i]])
        y_i, lower, higher = gp_ad.predict(x_i)
        lower_list.append(lower[0])
        higher_list.append(higher[0])
        # y_mean.append(y_i[0])
        # ratio = abs((higher[0] - lower[0]) / gp_ad.ppf / 2 / x_i.mean())

        percent_error = (y_i[0] - y_test[i]) / (y_test[i] + gp_ad.alpha)
        if (y_test[i] < lower[0] or y_test[i] > higher[0]) and abs(percent_error) > 0.01:
            abnormal = True
            index.append(i - start)
        else:
            abnormal = False
        print(i, y_test[i], y_i, lower, higher, abnormal, percent_error)  # x_i.std()

    x = numpy.array(range(0, end - start))
    # pyplot.plot(x, y_mean, color='g')
    pyplot.plot(x, y_target, color='b')
    pyplot.scatter(x[index], y_target[index], color='r', s=30, marker='*')
    pyplot.plot(x, y_smooth, color='g')
    pyplot.fill_between(x, lower_list, higher_list, alpha=0.2)

    pyplot.show()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    mock_data()
