#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : data_loader.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 2:21 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 2:21 下午 by shendu.ht  init
"""
import os

import numpy
import pandas

DATA_PATH = "example/data"


def get_project_path():
    """
    Get the absolute path of project AnomalyDetectionAlgorithmDev
    """
    cur_path = os.path.abspath(os.path.dirname(__file__))
    cur_path = cur_path.split(os.path.sep)

    if 'AnomalyDetectionAlgorithmDev' not in cur_path:
        raise NameError('Project error!')
    index = cur_path.index('AnomalyDetectionAlgorithmDev')
    return os.path.sep.join(cur_path[0: index + 1])


def get_scenes():
    """
    Get data scenes of project AnomalyDetectionAlgorithmDev
    """
    data_path = os.path.join(get_project_path(), DATA_PATH)
    return os.listdir(data_path)


def get_scene_file_names(scene):
    """
    Get the absolute data path list of definite scene of project AnomalyDetectionAlgorithmDev
    """
    scene_path = os.path.join(get_project_path(), DATA_PATH, scene)
    return os.listdir(scene_path)


def get_scene_data_path(scene, f_name):
    """
    Get data file's absolute paths.

    Args:
        scene: str
            Data scene, such as jitter
        f_name: str
            File name
    """
    data_path = os.path.join(get_project_path(), DATA_PATH)
    return os.path.join(data_path, scene, f_name)


def data_loader(scene=None, f_name=None):
    """
    Import data

    Args:
        scene: str
            Data file's relative path
        f_name: str
            File name

    Returns:
        numpy.ndarray
    """
    if scene is None:
        scene = get_scenes()[0]
    if f_name is None:
        f_name = get_scene_file_names(scene)[0]
    f_name = get_scene_data_path(scene, f_name)

    data = pandas.read_csv(f_name)
    if 'value' not in data.columns:
        raise ValueError('Metric not found!')

    # # set timestamp, not used here
    # if 'timestamp' in data.columns:
    #     timestamp = [datetime.datetime.fromtimestamp(i) for i in data['timestamp']]
    #     data['timestamp'] = timestamp
    #     data = data.set_index('timestamp')

    return numpy.asarray(data['value'])
