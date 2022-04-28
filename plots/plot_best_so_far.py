#!/usr/bin/python3


import csv
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, MaxNLocator)
import matplotlib.backends.backend_pdf


def main():
    parser = argparse.ArgumentParser('Best-so-far plot creator')
    parser.add_argument('--csv_paths', nargs=1, required=True,
                        help='Path to csv files with best-so-far data')
    args = parser.parse_args()
    path = args.csv_paths[0]
    plt.style.use('ggplot')
    figs = []
    opt_to_color = {'BO': 'red', 'LinearPCABO': 'blue', 'KernelPCABOInverse': 'green', 'pyCMA': 'purple'}
    opt_to_name = {'BO': 'BO', 'LinearPCABO': 'PCA-BO', 'KernelPCABOInverse': 'KernelPCABO', 'pyCMA': 'CMA'}
    for dim in [20, 40, 60]:
        N, M = 2, 5
        fig, axs = plt.subplots(N, M, figsize=(15, 5) )
        fig.supylabel(f'f - f* in {dim}D', x=0.07, fontsize=20)
        cnt = 0
        for fid in range(15, 25):
            for opt in ['BO', 'LinearPCABO', 'KernelPCABOInverse']:
                fname = f'{opt}_D{dim}_F{fid}'
                with open(os.path.join(path, fname), 'r') as f:
                    r = csv.reader(f, delimiter=' ')
                    next(r, None)
                    x, y, err = [], [], []
                    for row in r:
                        x.append(int(row[0]))
                        y.append(float(row[1]))
                        err.append(float(row[2]) / np.sqrt(float(row[3])))
                    x, y, err = np.array(x), np.array(y), np.array(err)
                    ax = axs[cnt // M, cnt % M]
                    if cnt // M == 0:
                        ax.get_xaxis().set_ticklabels([])
                    if dim == 60 or dim == 40:
                        ax.xaxis.set_major_locator(MultipleLocator(20))
                    else:
                        ax.xaxis.set_major_locator(MultipleLocator(10))
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    # ax.yaxis.set_major_locator(MultipleLocator(20))
                    ax.plot(x, y, c=opt_to_color[opt], linewidth=0.5, label=opt_to_name[opt])
                    ax.fill_between(x, y-err, y+err, facecolor=opt_to_color[opt], alpha=0.15)
                    ax.tick_params(axis='both', which='major', labelsize=9, pad=0)
                    ax.grid(which='both', linestyle='--', linewidth='0.05')
                    ax.set_facecolor('#ececec')
                    ax.set_title(f'F{fid}', loc='center', fontdict={'fontsize': 12, 'fontweight': 'medium'}, y=0.96)
            cnt += 1
        plt.subplots_adjust(wspace=0.18, hspace=0.15)
        # plt.margins(x=0, tight=True)
        if dim == 60:
            fig.supxlabel('iteration', fontsize=20)
        if dim == 20:
            handles, labels = ax.get_legend_handles_labels()
            leg = fig.legend(handles, labels, loc='upper center', ncol=4, prop={'size': 15})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(1.5)
        # plt.legend(bbox_to_anchor=(1.1, -1.1), bbox_transform=ax.transAxes)
        fig.savefig(f'convergence{dim}.pdf')
    # pdf = matplotlib.backends.backend_pdf.PdfPages("convergence.pdf")
    # for fig in figs:
        # pdf.savefig(fig)
    # pdf.close()


if __name__ == '__main__':
    main()
