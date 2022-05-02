#!/usr/bin/python3


import argparse
import sys
import shutil
import os
import csv
import math
import json
import functools
import ast
from zipfile import ZipFile
import io


class Description:
    def __init__(self, fid: int, dim: int, seed: int):
        self.fid = int(fid)
        self.dim = int(dim)
        self.seed = int(seed)

    def __eq__(self, other):
        if (isinstance(other, Description)):
            return self.fid == other.fid and self.dim == other.dim and self.seed == other.seed
        return False

    def __hash__(self):
        return hash((self.fid, self.dim, self.seed))

    def __str__(self):
        return f'F{self.fid}_D{self.dim}_Seed{self.seed}'


def process_stdout(stdout_file):
    with open(stdout_file, 'r') as f:
        lines = f.readlines()
        beg = lines[0].find('{')
        end = lines[0].rfind('}')
        m = json.loads(lines[0][beg:end+1].replace('\'', '\"'))
        result = Description(fid=m['fid'], dim=m['dim'], seed=m['seed'])
        doe_size, beg, cnt = 0, 0, 0
        for line in lines:
            if 'doe_size = ' in line:
                doe_size = int(line.split('=')[1])
            if 'DoE_value DoE_point' in line:
                beg = cnt + 1
                break
            cnt += 1
        doe = []
        for line in lines[beg:beg+doe_size]:
            value = float(line[:line.find('[')])
            x = ast.literal_eval(line[line.find('['):])
            x = [float(xi) for xi in x]
            doe.append((x, value))
        return result, doe


def get_best(doe):
    best_value, arg_best = float('inf'), []
    for x, v in doe:
        if v < best_value:
            best_value = v
            arg_best = x
    return (arg_best, best_value)


def add_logs_folder(folder_name):
    m = {}
    for f in os.listdir(folder_name):
        if f.endswith('.out'):
            description, doe = process_stdout(os.path.join(folder_name, f))
            m[description] = get_best(doe)
    return m


def main():
    parser = argparse.ArgumentParser('Extract DoE info')
    parser.add_argument('--log_paths', nargs='+', required=True,
                        help='Path to files with log from run in cluster')
    args = parser.parse_args()
    for log_path in args.log_paths:
        add_logs_folder(log_path)


if __name__ == '__main__':
    main()
