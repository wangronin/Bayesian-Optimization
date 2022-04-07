import sys
import os
import json
import datetime
from single_experiment import validate_optimizers


class ExperimentEnvironment:
    SLURM_SCRIPT_TEMPLATE = '''
#!/bin/env bash

#SBATCH --job-name=BO-##folder##
#SBATCH --array=0-##number##
#SBATCH --partition=cpu-long
#SBATCH --mem-per-cpu=1G
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=kirant9797@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=##logs_out##
#SBATCH --error=##logs_err##

FILES=(configs/*)
CONFIG=${FILES[$SLURM_ARRAY_TASK_ID]}
python ../single_experiment.py $CONFIG
'''

    def __init__(self):
        now = datetime.datetime.now()
        suffix = now.strftime('%d-%m-%Y_%Hh%Mm%Ss')
        folder_name = 'run_' + suffix
        os.makedirs(folder_name, exist_ok=False)
        print(f'Experiment root is: {folder_name}')
        self.experiment_root = os.path.abspath(folder_name)

    def set_up_by_experiment_config_file(self, experiment_config_file_name):
        self.__generate_configs(experiment_config_file_name)
        self.__generate_slurm_script()

    def __generate_slurm_script(self):
        slurm_script_file_name = os.path.join(self.experiment_root, 'slurm.sh')
        logs_folder = os.path.join(self.experiment_root, 'logs')
        logs_out = os.path.join(logs_folder, '%A_%a.out')
        logs_err = os.path.join(logs_folder, '%A_%a.err')
        script = ExperimentEnvironment.SLURM_SCRIPT_TEMPLATE\
                .replace('##folder##', self.result_folder_prefix)\
                .replace('##number##', str(self.generated_configs))\
                .replace('##logs_out##', logs_out)\
                .replace('##logs_err##', logs_err)
        with open(slurm_script_file_name, 'w') as f:
            f.write(script)

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

    def go_to_experiment_folder():
        os.chdir(self.experiment_root)


def main(argv):
    env = ExperimentEnvironment()
    env.set_up_by_experiment_config_file(argv[1])


if __name__ == '__main__':
    main(sys.argv)

