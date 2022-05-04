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
    best_so_far = []
    next(r, None)
    for row in r:
        best_so_far.append(float(row[2]))
    return best_so_far


def get_mean_sd(result_data, arrays):
    M = max(len(a) for a in arrays)
    N = len(arrays)
    mean = [0.] * M
    sd = [0.] * M
    counts = [0.] * M
    last = [0.] * N
    for j in range(M):
        cnt = 0
        for i in range(N):
            if j < len(arrays[i]):
                mean[j] += arrays[i][j]
                last[i] = arrays[i][j]
            else:
                mean[j] += last[i]
            cnt += 1
        mean[j] /= cnt
        counts[j] = cnt
        sd[j] = math.sqrt(sum((arrays[i][j] - mean[j]) **
                          2 for i in range(len(arrays)) if j < len(arrays[i])) / cnt)
    return mean, sd, counts


def debug_f23(arrays):
    import matplotlib.pyplot as plt
    for i in range(len(arrays)):
        fig = plt.figure()
        plt.plot([j for j in range(len(arrays[i]))], arrays[i])
        fig.savefig(f'{i}.png')


def process_cur_results(result_data, arrays, extract):
    file_fqn = os.path.join(
        extract, f'{result_data.opt}_D{result_data.dim}_F{result_data.fid}')
    mean, sd, cnt = get_mean_sd(result_data, arrays)
    with open(file_fqn, 'w') as f:
        f.write(f'runtime mean sd count\n')
        beg = 3 * result_data.dim if result_data.opt != 'pyCMA' else 1
        end = 5 * result_data.dim if result_data.opt != 'pyCMA' else 2 * result_data.dim
        step = 1
        for i in range(beg, end + step, step):
            if i-1 > len(mean):
                break
            f.write(f'{i} {mean[i-1]} {sd[i-1]} {cnt[i-1]}\n')


def process_cur_results_1(result_data, arrays, extract):
    file_fqn = os.path.join(
        extract, f'{result_data.opt}_D{result_data.dim}_F{result_data.fid}')
    mean, sd, cnt = get_mean_sd(result_data, arrays)
    with open(file_fqn, 'w') as f:
        f.write(f'runtime mean sd count\n')
        beg = 1
        end = 5 * result_data.dim
        step = 1
        for i in range(beg, end + step, step):
            if i-1 > len(mean):
                break
            f.write(f'{i} {mean[i-1]} {sd[i-1]} {cnt[i-1]}\n')


def main():
    parser = argparse.ArgumentParser('Processor of best-so-far')
    parser.add_argument('--zip_paths', nargs='+', required=True,
                        help='Path to zip file with ioh data to process')
    parser.add_argument('--extract', nargs=1, required=True,
                        help='Path to the folder where to extract all the data')
    args = parser.parse_args()
    extract_data_dir = args.extract[0]
    if os.path.exists(extract_data_dir) and not os.path.isdir(extract_data_dir):
        print(f'File {extract_data_dir} is not a directory')
    os.makedirs(extract_data_dir, exist_ok=True)
    gatherer = ResultsGatherer(read_result_callback=process_dat_file)
    for zip_path in args.zip_paths:
        gatherer.add_zip(zip_path)
    gatherer.process_results(process_result_callback=functools.partial(
        process_cur_results_1, extract=extract_data_dir))


if __name__ == '__main__':
    main()
