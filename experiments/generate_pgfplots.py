#!/usr/bin/python3


import sys
import shutil
import os
import csv
import math
import json


class PfgplotsChart:
    beg = '''
\\begin{tikzpicture}
    \\begin{axis}[
        width=0.5\\textwidth,
     ]
'''
    end = '''     \\end{axis}
\\end{tikzpicture}
'''

    def __init__(self):
        self.files = []

    def add_file(self, file_name):
        self.files.append(file_name)

    def generate_tex(self):
        if not self.files:
            return ''
        code = [PfgplotsChart.beg]
        for file in self.files:
            s = '        \\addplot+[color=blue, mark=none, error bars/.cd, y dir=both, y explicit] table[x index=0,y index=1,y error index=2] {data/' + file + '};\n'
            code.append(s)
        code.append(PfgplotsChart.end)
        return ''.join(code)



def main(argv):
    experiment_config_file_name = argv[1]
    with open(experiment_config_file_name, 'r') as f:
        config = json.load(f)
    fids = config['fids']
    iids = config['iids']
    dims = config['dims']
    reps = config['reps']
    optimizers = config['optimizers']
    lb, ub = config['lb'], config['ub']
    runs_number = len(optimizers) * len(fids) * len(iids) * len(dims) * reps
    cur_config_number = 0
    for fid in fids:
        for iid in iids:
            for dim in dims:
                dirs_to_process = []
                pgfchart = PfgplotsChart()
                for my_optimizer_name in optimizers:
                    file_name = f'{my_optimizer_name}-D{dim}-F{fid}.csv'
                    if os.path.exists(os.path.join('processed', file_name)):
                        pgfchart.add_file(file_name)
                print(pgfchart.generate_tex())


if __name__ == '__main__':
    main(sys.argv)
