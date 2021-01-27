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

import numpy
from matplotlib import pyplot

from algorithm.supervised.gaussian_process_detector import GaussianProcessAD
from data_preparation.cleaning.transform import slide_window
from example.algorithm.data_loader import data_loader


def main():
    original_data = data_loader('jitter', 'ja_set_online.csv')
    window_list = [5, 10, 15, 20, 30]
    test_len = 360

    print(','.join(['window_size', 'sample size', 'test round', 'mape']))

    for window in window_list:
        x_n, y_n = slide_window(original_data, window)
        x_train, y_train = x_n[:-test_len, :], y_n[:-test_len]
        x_test, y_test = x_n[-test_len:, :], y_n[-test_len:]

        sample_size_list = [100, 200]
        for sample_size in sample_size_list:
            for i in range(100):
                index = [random.randint(0, x_train.shape[0] - 1) for _ in range(sample_size)]
                gp_ad = GaussianProcessAD(x_train[index], y_train[index], n_iter=10)
                gp_ad.fit()
                ape, mape = gp_ad.evaluate(x_test, y_test)
                print(','.join([str(window), str(sample_size), str(i), str(round(mape, 4))]))
    return


def mock_data():
    original_data = data_loader('jitter', 'ja_set_online.csv')
    window = 15
    x_n, y_n = slide_window(original_data, window)

    # pyplot.plot(original_data)
    # pyplot.show()

    abnormal = False
    y_i = None
    y_mean = []
    lower_list = []
    higher_list = []

    start = 2100
    end = 2200
    y_target = y_n[start: end]

    for i in range(start, end):
        # if y_i is not None and abnormal:
        # print(original_data[i - window: i + window], y_i)
        # original_data[i + window] = y_i[0]
        # x_n, y_n = slide_window(original_data, window)

        # x_train = numpy.concatenate((x_n[i - 100:i], x_n[i - 1440 - 100:i - 1440]), axis=0)
        # y_train = numpy.concatenate((y_n[i - 100:i], y_n[i - 1440 - 100:i - 1440]), axis=0)
        x_train = x_n[i - 200:i]
        y_train = y_n[i - 200:i]

        gp_ad = GaussianProcessAD(x_train, y_train, p=0.0001, n_iter=10)
        gp_ad.fit()

        x_i = numpy.array([x_n[i]])
        y_i, lower, higher = gp_ad.predict(x_i)
        lower_list.append(lower[0])
        higher_list.append(higher[0])
        y_mean.append(y_i[0])
        if y_n[i] < lower[0] or y_n[i] > higher[0] or (
                (higher[0] - lower[0]) / gp_ad.ppf / 2 > x_i.std() * gp_ad.ppf and x_i.std() / x_i.mean() > 1e-3):
            abnormal = True
        else:
            abnormal = False
        print(i, y_n[i], y_i, lower, higher, abnormal, x_i.std())

    x = list(range(0, end - start))
    pyplot.plot(x, y_mean, color='g')
    pyplot.plot(x, y_target, color='r')
    pyplot.fill_between(x, lower_list, higher_list, alpha=0.2)

    pyplot.show()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    mock_data()
