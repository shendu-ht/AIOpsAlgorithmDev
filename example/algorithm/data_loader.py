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
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

DATA_PATH = "example/data"


class DataLoader(ABC):
    """
    数据获取的基类
    """

    @abstractmethod
    def __init__(self):
        """
        Initial
        """
        self.project_path = ''

    @abstractmethod
    def set_project_path(self):
        """
        Get the absolute path of project AnomalyDetectionAlgorithmDev.
        """
        cur_path = os.path.abspath(os.path.dirname(__file__))
        cur_path = cur_path.split(os.path.sep)

        if 'AnomalyDetectionAlgorithmDev' not in cur_path:
            raise NameError('Project error!')
        index = cur_path.index('AnomalyDetectionAlgorithmDev')
        self.project_path = os.path.sep.join(cur_path[0: index + 1])

    @abstractmethod
    def get_scenes(self):
        """
        Get data scenes of project AnomalyDetectionAlgorithmDev.
        """
        if self.project_path == '':
            self.set_project_path()
        data_path = os.path.join(self.project_path, DATA_PATH)
        return os.listdir(data_path)

    @abstractmethod
    def get_scene_file_names(self, scene):
        """
        Get the absolute data path list of definite scene of project AnomalyDetectionAlgorithmDev
        """

        if self.project_path == '':
            self.set_project_path()
        scene_path = os.path.join(self.project_path, DATA_PATH, scene)
        return os.listdir(scene_path)

    @abstractmethod
    def get_scene_data_path(self, scene, f_name):
        """
        Get data file's absolute paths.
        """

        if self.project_path == '':
            self.set_project_path()
        data_path = os.path.join(self.project_path, DATA_PATH)
        return os.path.join(data_path, scene, f_name)

    @abstractmethod
    def data_loader(self, scene=None, f_name=None):
        """
        Import data
        """
        pass


class JitterCsvDataLoader(DataLoader):
    """
    Jitter Project Data Loader
    """

    def __init__(self):
        super().__init__()
        self.scene = 'jitter'

    def set_project_path(self):
        super().set_project_path()

    def get_scenes(self):
        return super().get_scenes()

    def get_scene_file_names(self, scene):
        return super().get_scene_file_names(scene=scene)

    def get_scene_data_path(self, scene, f_name):
        return super().get_scene_data_path(scene=scene, f_name=f_name)

    def data_loader(self, scene=None, f_name=None):
        """
        Import Data.

        Parameters
        ----------
        scene: str
            Scene name, Default `jitter`.
        f_name: str
            File name.

        Returns
        -------
        np.ndarray, shape=(n, )
            Series Data

        """
        if scene is None:
            scene = self.scene
        if f_name is None:
            f_name = self.get_scene_file_names(scene)[0]
        f_name = self.get_scene_data_path(scene, f_name)

        data = pd.read_csv(f_name)
        if 'value' not in data.columns:
            raise ValueError('Metric not found!')

        return np.asarray(data['value'])


class UCRTimeSeriesDataLoader(DataLoader):
    """
    UCR Time Series Classification Archive.

    References:
        Chen Y, Keogh E, Hu B, Begum N, Bagnall A, Mueen A, Batista G (2015) The UCR time series classification archive.
        URL: http://www.cs.ucr.edu/~eamonn/time_series_data/
    """

    def __init__(self):
        super().__init__()
        self.scene = 'UCR_Time_Series'

    def set_project_path(self):
        super().set_project_path()

    def get_scenes(self):
        return super().get_scenes()

    def get_scene_file_names(self, scene):
        return super().get_scene_file_names(scene=scene)

    def get_scene_data_path(self, scene, f_name):
        return super().get_scene_data_path(scene=scene, f_name=f_name)

    def data_loader(self, scene=None, f_name=None):
        """
        Import Data.

        Parameters
        ----------
        scene: str
            Scene name. Two layer, the first is default `UCR_Time_Series`, the second is scene detail.
        f_name: str
            File name.

        Returns
        -------
        np.ndarray, shape=(n, m)
            Series input, n is the sample size, and m is the series length.
        np.ndarray, shape=(n, )
            Series label

        """
        if scene is None:
            scene_child = self.get_scene_file_names(self.scene)[0]
            scene = os.path.join(self.scene, scene_child)
        if f_name is None:
            f_name = self.get_scene_file_names(scene)[0]
        f_name = self.get_scene_data_path(scene, f_name)
        data = pd.read_csv(f_name, header=None)

        x, y = [], []
        for i in data.index:
            data_i = data.loc[i]
            label = data_i[0]
            data_detail = data_i[1:]
            x.append(np.asarray(data_detail))
            y.append(label)
        return np.asarray(x), np.asarray(y)
