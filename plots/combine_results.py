import argparse
import sys
import shutil
import os
import csv
import math
import json
import functools
from zipfile import ZipFile
import io
from experiments.my_logger import MyIOHFormatOnEveryEvaluationLogger as logger


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
        self.info_processed = {}
        self.dat_processed = {}

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

    def __add_result(self, processed, result, result_data):
        processed[result] = result_data

    def add_zip(self, path_to_ioh_zip):
        with ZipFile(path_to_ioh_zip) as myzip:
            for name in myzip.namelist():
                if name.endswith('.dat') and self.dat_callback:
                    parts = name.split('/')
                    dir_name = parts[0]
                    with myzip.open(name, 'r') as dat_file:
                        f = io.TextIOWrapper(dat_file)
                        r = csv.reader(f, delimiter=' ')
                        result_data = self.dat_callback(r)
                        self.__add_result(self.dat_processed,
                                          dir_name, result_data)
                elif name.endswith('.info') and self.info_callback:
                    parts = name.split('/')
                    dir_name = parts[0]
                    with myzip.open(name, 'r') as dat_file:
                        f = io.TextIOWrapper(dat_file)
                        r = f.readlines()
                        result_data = self.info_callback(r)
                        self.__add_result(self.info_processed,
                                          dir_name, result_data)

    def process_results(self, process_result_callback):
        processed_dat_info = {}
        for name, dat in self.dat_processed.items():
            info = self.info_processed[name]
            parts = name.split('/')
            dir_name = parts[0]
            opt = self.__get_optimizer(dir_name)
            fid = self.__get_fid(dir_name)
            dim = self.__get_dim(dir_name)
            res = Result(fid=fid, dim=dim, opt=opt)
            if res in processed_dat_info:
                processed_dat_info[res].append((dat, info))
            else:
                processed_dat_info[res] = [(dat, info)]
        for name, dat_info in processed_dat_info.items():
            process_result_callback(name, dat_info)


def process_file(r):
    rows = []
    next(r, None)
    for row in r:
        rows.append(row)
    return rows


my_alg_to_normal_name = {'BO': 'BO',
                         'LinearPCABO': 'PCA-BO',
                         'KernelPCABOInverse': 'KPCA-BO',
                         'pyCMA': 'CMA-ES'}


def find_time_in_info(info):
    for l in info:
        if '|' in l:
            d = l.split('|')
            return float(d[1])


def process_cur_results(result_data, arrays, extract):
    os.chdir(extract)
    for (dat, info) in arrays:
        l = logger(folder_name=f'{my_alg_to_normal_name[result_data.opt]}_D{result_data.dim}_F{result_data.fid}',
                   algorithm_name=my_alg_to_normal_name[result_data.opt])
        l._set_up_logger(result_data.fid, 0, result_data.dim, 'func_name')
        for row in dat:
            evaluation_cnt = int(row[0])
            cur_fitness = float(row[1])
            best_so_far = float(row[2])
            l.log(evaluation_cnt, cur_fitness, best_so_far)
        time = find_time_in_info(info)
        l.finish_logging(time)
    os.chdir('..')


def main():
    parser = argparse.ArgumentParser('Combine and prettify results')
    parser.add_argument('--zip_paths', nargs='+', required=True,
                        help='Path to zip file with ioh data to process')
    parser.add_argument('--extract', nargs=1, required=True,
                        help='Path to the folder where to put the resulting archives')
    args = parser.parse_args()
    extract_data_dir = args.extract[0]
    if os.path.exists(extract_data_dir) and not os.path.isdir(extract_data_dir):
        print(f'File {extract_data_dir} is not a directory')
    os.makedirs(extract_data_dir, exist_ok=True)
    gatherer = ResultsGatherer(
        read_result_callback=process_file, read_info_callback=lambda x: x)
    for zip_path in args.zip_paths:
        gatherer.add_zip(zip_path)
    gatherer.process_results(process_result_callback=functools.partial(
        process_cur_results, extract=extract_data_dir))


if __name__ == '__main__':
    main()
