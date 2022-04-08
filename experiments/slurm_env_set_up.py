import sys
import os
import json
import datetime
from single_experiment import validate_optimizers


class ExperimentEnvironment:
    SLURM_SCRIPT_TEMPLATE = '''#!/bin/env bash

#SBATCH --job-name=##folder##
#SBATCH --array=##from_number##-##to_number##
#SBATCH --partition=cpu-long
#SBATCH --mem-per-cpu=1G
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=kirant9797@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=##logs_out##
#SBATCH --error=##logs_err##

python ../single_experiment.py configs/${SLURM_ARRAY_TASK_ID}.json
'''

    def __init__(self):
        now = datetime.datetime.now()
        suffix = now.strftime('%d-%m-%Y_%Hh%Mm%Ss')
        folder_name = 'run_' + suffix
        os.makedirs(folder_name, exist_ok=False)
        print(f'Experiment root is: {folder_name}')
        self.experiment_root = os.path.abspath(folder_name)
        self.__max_array_size = 1000
        self.__number_of_slurm_scripts = 0

    def set_up_by_experiment_config_file(self, experiment_config_file_name):
        self.__generate_configs(experiment_config_file_name)
        self.__create_log_dir()
        self.__generate_slurm_script()

    def __create_log_dir(self):
        self.logs_folder = os.path.join(self.experiment_root, 'logs')
        os.mkdir(self.logs_folder)

    def __generate_slurm_script(self):
        self.__number_of_slurm_scripts = 0
        logs_out = os.path.join(self.logs_folder, '%A_%a.out')
        logs_err = os.path.join(self.logs_folder, '%A_%a.err')
        script = ExperimentEnvironment.SLURM_SCRIPT_TEMPLATE\
                .replace('##folder##', self.result_folder_prefix)\
                .replace('##logs_out##', logs_out)\
                .replace('##logs_err##', logs_err)
        offset = 0
        for i in range(self.generated_configs // self.__max_array_size):
            with open(os.path.join(self.experiment_root, f'slurm{self.__number_of_slurm_scripts}.sh'), 'w') as f:
                f.write(script\
                        .replace('##from_number##', str(offset))\
                        .replace('##to_number##', str(offset + self.__max_array_size - 1)))
            offset += self.__max_array_size
            self.__number_of_slurm_scripts += 1
        r = self.generated_configs % self.__max_array_size
        if r > 0:
            with open(os.path.join(self.experiment_root, f'slurm{self.__number_of_slurm_scripts}.sh'), 'w') as f:
                f.write(script\
                        .replace('##from_number##', str(offset))\
                        .replace('##to_number##', str(offset + r - 1)))
            offset += self.__max_array_size
            self.__number_of_slurm_scripts += 1

    def __generate_configs(self, experiment_config_file_name):
        with open(experiment_config_file_name, 'r') as f:
            config = json.load(f)
        self.result_folder_prefix = config['folder']
        fids = config['fids']
        iids = config['iids']
        dims = config['dims']
        reps = config['reps']
        if 'extra' not in config.keys():
            config['extra'] = ''
        optimizers = config['optimizers']
        lb, ub = config['lb'], config['ub']
        validate_optimizers(optimizers)
        runs_number = len(optimizers) * len(fids) * len(iids) * len(dims) * reps
        cur_config_number = 0
        configs_dir = os.path.join(self.experiment_root, 'configs')
        os.makedirs(configs_dir, exist_ok=False)
        with open(os.path.join(self.experiment_root, 'description.json'), 'w') as f:
            json.dump(config, f, indent=4)
        for my_optimizer_name in optimizers:
            for fid in fids:
                for iid in iids:
                    for dim in dims:
                        # print(f'Ids for opt={my_optimizer_name}, fid={fid}, iid={iid}, dim={dim} are [{cur_config_number}, {cur_config_number+reps-1}]')
                        for rep in range(reps):
                            experiment_config = {
                                    'folder': f'{self.result_folder_prefix}_Opt-{my_optimizer_name}_F-{fid}_Dim-{dim}_Rep-{rep}_Id-{cur_config_number}',
                                    'opt': my_optimizer_name,
                                    'fid': fid,
                                    'iid': iid,
                                    'dim': dim,
                                    'seed': rep,
                                    'lb': lb,
                                    'ub': ub,
                                    }
                            cur_config_file_name = f'{cur_config_number}.json'
                            with open(os.path.join(configs_dir, cur_config_file_name), 'w') as f:
                                json.dump(experiment_config, f)
                            cur_config_number += 1
        print(f'Generated {cur_config_number} files')
        self.generated_configs = cur_config_number

    def print_helper(self):
        print(f'cd {self.experiment_root} && for (( i=0; i<{self.__number_of_slurm_scripts}; ++i )); do sbatch slurm$i.sh; done')


def main(argv):
    env = ExperimentEnvironment()
    env.set_up_by_experiment_config_file(argv[1])
    env.print_helper()


if __name__ == '__main__':
    main(sys.argv)

