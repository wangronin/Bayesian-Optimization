import sys
import shutil
import os
import csv
import math
import json
import argparse
from zipfile import ZipFile
import io


class Result:
    def __init__(self, fid, dim, opt):
        self.fid = fid
        self.dim = dim
        self.opt = opt

    def __eq__(self, other):
        if (isinstance(other, Result)):
            return self.fid == other.fid and self.dim == other.dim and self.opt == other.opt
        return False

    def __hash__(self):
        return hash((self.fid, self.dim, self.opt))


class ResultsGatherer:
    def __init__(self, read_result_callback=None, read_info_callback=None):
        self.processed = {}
        self.dat_callback = read_result_callback
        self.info_callback = read_info_callback

    def __find_name_in_dirname(self, dir_name, name):
        for p in dir_name.split('_'):
            if name in p:
                return p.split('-')[1]
        return None

    def __get_optimizer(self, dir_name):
        return self.__find_name_in_dirname(dir_name, 'Opt')

    def __get_fid(self, dir_name):
        return self.__find_name_in_dirname(dir_name, 'F')

    def __get_dim(self, dir_name):
        return int(self.__find_name_in_dirname(dir_name, 'Dim'))

    def __add_result(self, result, result_data):
        if result in self.processed.keys():
            self.processed[result].append(result_data)
        else:
            self.processed[result] = [result_data]

    def add_zip(self, path_to_ioh_zip):
        with ZipFile(path_to_ioh_zip) as myzip:
            for name in myzip.namelist():
                if name.endswith('.dat') and self.dat_callback:
                    parts = name.split('/')
                    dir_name = parts[0]
                    opt = self.__get_optimizer(dir_name)
                    fid = self.__get_fid(dir_name)
                    dim = self.__get_dim(dir_name)
                    with myzip.open(name, 'r') as dat_file:
                        f = io.TextIOWrapper(dat_file)
                        r = csv.reader(f, delimiter=' ')
                        result_data = self.dat_callback(r)
                        self.__add_result(
                            Result(opt=opt, fid=fid, dim=dim), result_data)
                elif name.endswith('.info') and self.info_callback:
                    parts = name.split('/')
                    dir_name = parts[0]
                    opt = self.__get_optimizer(dir_name)
                    fid = self.__get_fid(dir_name)
                    dim = self.__get_dim(dir_name)
                    with myzip.open(name, 'r') as dat_file:
                        f = io.TextIOWrapper(dat_file)
                        r = f.readlines()
                        result_data = self.info_callback(r)
                        self.__add_result(
                            Result(opt=opt, fid=fid, dim=dim), result_data)

    def process_results(self, process_result_callback):
        for k, v in self.processed.items():
            process_result_callback(k, v)
