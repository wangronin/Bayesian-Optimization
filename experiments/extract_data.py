#!/usr/bin/python3


import sys
import shutil
import os
import csv
import math
import json


def process_dat_file(dat_file):
    best_so_far = []
    with open(dat_file, 'r') as f:
        r = csv.reader(f, delimiter=' ')
        next(r, None)
        for row in r:
            best_so_far.append(float(row[2]))
    return best_so_far


def process_folder(folder_name):
    for root, d_names, f_names in os.walk(folder_name):
        for file in f_names:
            file_path = os.path.join(root, file)
            if file.startswith('IOH') and file.endswith('.dat'):
                return process_dat_file(file_path)


def get_mean_sd(arrays):
    mean = [0.] * len(arrays[0])
    sd = [0.] * len(arrays[0])
    for j in range(len(arrays[0])):
        for i in range(len(arrays)):
            mean[j] += arrays[i][j]
        mean[j] /= len(arrays)
        sd[j] = math.sqrt(sum((arrays[i][j] - mean[j]) **
                          2 for i in range(len(arrays))) / len(arrays))
    return mean, sd


processed = 'processed'
have = {}


def process_directories(dirs, alg, dim, fun):
    arrays = []
    for d in dirs:
        array = process_folder(d)
        if array is not None:
            arrays.append(array)
    if not arrays:
        have[(alg, dim, fun)] = False
        return
    have[(alg, dim, fun)] = True
    mean, sd = get_mean_sd(arrays)
    file_name = f'{alg}-D{dim}-F{fun}.csv'
    with open(os.path.join(processed, file_name), 'w') as f:
        f.write('runtime mean sd\n')
        for i in range(len(mean)):
            f.write(f'{i+1} {mean[i]} {sd[i]}\n')


def main(argv):
    experiment_config_file_name = argv[1]
    with open(experiment_config_file_name, 'r') as f:
        config = json.load(f)
    result_folder_prefix = config['folder']
    fids = config['fids']
    iids = config['iids']
    dims = config['dims']
    reps = config['reps']
    optimizers = config['optimizers']
    lb, ub = config['lb'], config['ub']
    runs_number = len(optimizers) * len(fids) * len(iids) * len(dims) * reps
    cur_config_number = 0
    os.makedirs(processed, exist_ok=False)
    for my_optimizer_name in optimizers:
        for fid in fids:
            for iid in iids:
                for dim in dims:
                    dirs_to_process = []
                    for rep in range(reps):
                        cur_dir = f'{result_folder_prefix}_Opt-{my_optimizer_name}_F-{fid}_Dim-{dim}_Rep-{rep}_Id-{cur_config_number}-0'
                        cur_config_number += 1
                        dirs_to_process.append(cur_dir)
                    process_directories(
                        dirs_to_process, my_optimizer_name, dim, fid)


if __name__ == '__main__':
    main(sys.argv)
