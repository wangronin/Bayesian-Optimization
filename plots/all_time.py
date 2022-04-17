#!/usr/bin/python3


import argparse
import sys
import shutil
import os
import csv
import math
import json
from iohhandler import ResultsGatherer
import functools


def process_dat_file(r):
    for l in r:
        if '|' in l:
            d = l.split('|')
            return [float(d[1])]


def get_mean_sd(arrays):
    M = max(len(a) for a in arrays)
    N = len(arrays)
    mean = [0.] * M
    sd = [0.] * M
    for j in range(M):
        cnt = 0
        for i in range(N):
            if j < len(arrays[i]):
                mean[j] += arrays[i][j]
                cnt += 1
        mean[j] /= cnt
        sd[j] = math.sqrt(sum((arrays[i][j] - mean[j]) **
                          2 for i in range(len(arrays)) if j < len(arrays[i])) / cnt)
    return mean, sd


def process_cur_results(result_data, arrays, extract):
    file_fqn = os.path.join(extract, f'{result_data.opt}_D{result_data.dim}_F{result_data.fid}')
    mean, sd = get_mean_sd(arrays)
    with open(file_fqn, 'w') as f:
        f.write(f'mean_cpu_time sd\n')
        f.write(f'{mean[0]} {sd[0]}\n')


def main():
    parser = argparse.ArgumentParser('Processor of cpu time')
    parser.add_argument('--zip_paths', nargs='+', required=True, help='Path to zip file with ioh data to process')
    parser.add_argument('--extract', nargs=1, required=True, help='Path to the folder where to extract all the data')
    args = parser.parse_args()
    extract_data_dir = args.extract[0]
    if os.path.exists(extract_data_dir) and not os.path.isdir(extract_data_dir):
        print(f'File {extract_data_dir} is not a directory')
    os.makedirs(extract_data_dir, exist_ok=True)
    gatherer = ResultsGatherer(read_info_callback=process_dat_file)
    for zip_path in args.zip_paths:
        gatherer.add_zip(zip_path)
    gatherer.process_results(process_result_callback=functools.partial(process_cur_results, extract=extract_data_dir))


if __name__ == '__main__':
    main()
