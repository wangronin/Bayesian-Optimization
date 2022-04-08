#!/usr/bin/python3

import sys
import os


def remove_kernel_info(dat_file_path, output_file):
    with open(dat_file_path, 'r') as f, open(output_file, 'w') as of:
        for l in f.readlines():
            output_l = ''
            for i in range(len(l)):
                if l[i] == '{':
                    output_l += 'None\n'
                    break
                else:
                    output_l += l[i]
            of.write(output_l)


def cat(from_file, to_file):
    with open(to_file, 'w') as outfile:
        with open(from_file) as infile:
            for line in infile:
                outfile.write(line)


def process_file(dat_file):
    tmp_file_path = os.path.join(os.path.dirname(dat_file), 'tmp')
    remove_kernel_info(dat_file, tmp_file_path)
    cat(tmp_file_path, dat_file)
    os.remove(tmp_file_path)


incomplete_results = []


def process_info_file(info_file):
    with open(info_file, 'r') as f:
        if len(f.readlines()) < 2:
            incomplete_results.append(os.path.dirname(info_file))


def main(argv):
    result_folder_name = argv[1]
    d = '.'
    resuls_folders = [os.path.join(d, o) for o in os.listdir(d)
                      if os.path.isdir(os.path.join(d, o)) and o.startswith(result_folder_name)]
    for results_folder in resuls_folders:
        for root, d_names, f_names in os.walk(results_folder):
            for file in f_names:
                if file.startswith('IOH') and file.endswith('.info'):
                    process_info_file(os.path.join(root, file))

    import shutil

    for d in incomplete_results:
        print(d)
        shutil.rmtree(d)


if __name__ == '__main__':
    main(sys.argv)
